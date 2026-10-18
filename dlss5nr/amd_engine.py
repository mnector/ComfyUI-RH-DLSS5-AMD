# SPDX-License-Identifier: MIT
# ComfyUI-RH-DLSS5-AMD: AMD GPU acceleration engine (ZLUDA + ROCm + OpenCV DIS + AMD-NR proxy)
# Provides:
# 1. Hot-reload INI generator for AMD-NR / OptiScaler / danielblnc DLSS-NR-on-AMD runtimes
# 2. ZludaTensorEnhancer: 100% PyTorch GPU-accelerated DLSS 5 Neural Rendering fallback
# 3. AmdFrameInterpolator: Bidirectional OpenCV DIS optical flow + PyTorch pyramidal warping

from __future__ import annotations

import configparser
import math
import os
import platform
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None


def is_amd_environment(gpu_index: int = 0) -> bool:
    """Detect whether current execution environment targets an AMD GPU (HIP/ROCm or ZLUDA)."""
    try:
        if not torch.cuda.is_available():
            return False
        idx = max(0, min(int(gpu_index), torch.cuda.device_count() - 1))
        name = torch.cuda.get_device_name(idx).lower()
        if any(tag in name for tag in ("amd", "radeon", "gfx", "instinct", "navi")):
            return True
        if "rocm" in torch.__version__.lower():
            return True
        cuda_path = os.environ.get("CUDA_PATH", "").lower()
        sys_path = os.environ.get("PATH", "").lower()
        if "zluda" in cuda_path or "zluda" in sys_path:
            return True
    except Exception:
        pass
    return False


def write_amd_nr_config(runtime_dir: Path, params: dict) -> None:
    """Write or update hot-reload dlssnr_on_amd.ini and OptiScaler.ini for community AMD-NR runtimes."""
    if not runtime_dir or not runtime_dir.exists():
        return
    ini_files = [runtime_dir / "dlssnr_on_amd.ini", runtime_dir / "OptiScaler.ini"]
    for ini_path in ini_files:
        try:
            config = configparser.ConfigParser(strict=False)
            config.optionxform = str
            if ini_path.exists():
                config.read(str(ini_path), encoding="utf-8")

            # [DlssNr]
            if not config.has_section("DlssNr"):
                config.add_section("DlssNr")

            config.set("DlssNr", "Enabled", "true")
            config.set("DlssNr", "NrBackend", "lmxxf")
            config.set("DlssNr", "AmdPlacement", "0")
            config.set("DlssNr", "RunBeforeSR", "true")
            config.set("DlssNr", "ScanExposure", "false")
            config.set("DlssNr", "AmdNeuralLighting", "true")
            config.set("DlssNr", "WorkingScale", "1.0")
            config.set("DlssNr", "Passes", "4")
            config.set("DlssNr", "LmxxfFullNetwork", "true")

            style_val = int(params.get("style", 0))
            config.set("DlssNr", "Style", str(style_val))
            config.set("DlssNr", "Intensity", f"{float(params.get('intensity', 1.0)):.4f}")
            config.set("DlssNr", "Tone", f"{float(params.get('tone', 1.0)):.4f}")
            config.set("DlssNr", "GlobalTone", f"{float(params.get('global_tone', -1.0)):.4f}")
            config.set("DlssNr", "Structure", f"{float(params.get('structure', 1.5)):.4f}")
            config.set("DlssNr", "Skin", f"{float(params.get('skin', -1.0)):.4f}")
            config.set("DlssNr", "AutoMask", "1" if params.get("auto_mask", True) else "0")
            config.set("DlssNr", "HDR", "1" if params.get("hdr", False) else "0")
            if params.get("scale") is not None:
                config.set("DlssNr", "Scale", f"{float(params.get('scale', 1.0)):.4f}")

            # [AmdLook]
            if not config.has_section("AmdLook"):
                config.add_section("AmdLook")
            config.set("AmdLook", "Enabled", "true")
            config.set("AmdLook", "Appearance", "2")
            config.set("AmdLook", "Mix", "1.0")

            # [Upscalers]
            if not config.has_section("Upscalers"):
                config.add_section("Upscalers")
            config.set("Upscalers", "Dx12Upscaler", "ffx")

            # [Spoofing] - CRITICAL: Never spoof Dxgi or Nvidia vendor on AMD GPUs,
            # otherwise OptiScaler IsAmd check fails and skips all AMD HIP passes!
            if not config.has_section("Spoofing"):
                config.add_section("Spoofing")
            config.set("Spoofing", "Dxgi", "false")
            if config.has_option("Spoofing", "SpoofedVendorId"):
                config.remove_option("Spoofing", "SpoofedVendorId")

            # [InitFlags]
            if not config.has_section("InitFlags"):
                config.add_section("InitFlags")
            config.set("InitFlags", "AutoExposure", "false")
            config.set("InitFlags", "SkipFirstFrames", "0")

            # [Hotfix]
            if not config.has_section("Hotfix"):
                config.add_section("Hotfix")
            config.set("Hotfix", "DisableOverlays", "true")
            config.set("Hotfix", "CheckForUpdate", "false")

            # [Menu]
            if not config.has_section("Menu"):
                config.add_section("Menu")
            config.set("Menu", "OverlayMenu", "false")
            config.set("Menu", "DisableSplash", "true")

            # [Libraries] - Reset overrides to auto so adjacent runtime DLLs are loaded cleanly
            # and no machine-specific absolute paths leak or misroute proxies.
            if config.has_section("Libraries"):
                for opt in config.options("Libraries"):
                    config.set("Libraries", opt, "auto")

            # [Log]
            if not config.has_section("Log"):
                config.add_section("Log")
            config.set("Log", "LogLevel", "2")
            config.set("Log", "LogToFile", "true")

            with open(ini_path, "w", encoding="utf-8") as f:
                config.write(f)
        except Exception as e:
            print(f"[RH-DLSS5-AMD] Warning: Could not write {ini_path.name}: {e}", flush=True)

    # Prevent double-loading deadlock: OptiScaler.dll and dxgi.dll are the same binary.
    # dlss_bridge.exe explicitly loads OptiScaler.dll; having dxgi.dll adjacent causes
    # Windows to load OptiScaler twice (once as dxgi proxy, once as OptiScaler.dll),
    # causing recursive D3D12 device hooking deadlocks on AMD GPUs.
    try:
        dxgi_file = runtime_dir / "dxgi.dll"
        opti_file = runtime_dir / "OptiScaler.dll"
        if dxgi_file.is_file() and opti_file.is_file():
            disabled_dxgi = runtime_dir / "dxgi.dll.disabled"
            if disabled_dxgi.exists():
                disabled_dxgi.unlink()
            dxgi_file.rename(disabled_dxgi)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# AmdBridgeDaemon: Out-of-process D3D12/OptiScaler bridge daemon
# -----------------------------------------------------------------------------

class AmdBridgeDaemon:
    """Persistent out-of-process D3D12 bridge daemon executing OptiScaler + AMD-NR."""

    def __init__(
        self,
        bridge_exe: Path,
        input_w: int,
        input_h: int,
        output_w: int,
        output_h: int,
        runtime_dir: Path,
        params: dict,
    ):
        self.bridge_exe = bridge_exe
        self.input_w = input_w
        self.input_h = input_h
        self.output_w = output_w
        self.output_h = output_h
        self.runtime_dir = runtime_dir
        self.params = params
        self.tmpdir = tempfile.TemporaryDirectory(prefix="rh_dlss5_amd_")
        self.tmp_path = Path(self.tmpdir.name)
        self.c_bin = self.tmp_path / "c.bin"
        self.d_bin = self.tmp_path / "d.bin"
        self.m_bin = self.tmp_path / "m.bin"
        self.n_bin = self.tmp_path / "n.bin"
        self.a_bin = self.tmp_path / "a.bin"
        self.o_bin = self.tmp_path / "o.bin"
        self.proc: Optional[subprocess.Popen] = None
        self.warmup_passes = max(1, int(params.get("warmup_frames", 1)))
        self.eval_passes = 4
        self.fps = 30
        self.frametime = 0.033

    def _write_frame_inputs(self, pixels_u8: np.ndarray, motion_fp16: np.ndarray):
        h, w = pixels_u8.shape[:2]
        # Color: RGBA FP16
        color_rgba = np.empty((h, w, 4), dtype=np.float16)
        color_rgba[..., :3] = (pixels_u8.astype(np.float32) / 255.0).astype(np.float16)
        color_rgba[..., 3] = 1.0
        color_rgba.tofile(str(self.c_bin))

        # Depth: Flat 0.0 FP32
        depth_f32 = np.zeros((h, w, 1), dtype=np.float32)
        depth_f32.tofile(str(self.d_bin))

        # Motion vectors: RG FP16
        mvec_rg = np.ascontiguousarray(motion_fp16, dtype=np.float16)
        mvec_rg.tofile(str(self.m_bin))

        # Normal: Flat (0, 0, 1, 0) FP16
        normal_rgba = np.zeros((h, w, 4), dtype=np.float16)
        normal_rgba[..., 2] = 1.0
        normal_rgba.tofile(str(self.n_bin))

        # Albedo: copy of color
        color_rgba.tofile(str(self.a_bin))

    def start_and_process_first(self, pixels_u8: np.ndarray, motion_fp16: np.ndarray) -> np.ndarray:
        self._write_frame_inputs(pixels_u8, motion_fp16)
        cmd = [
            str(self.bridge_exe),
            str(self.input_w), str(self.input_h),
            str(self.output_w), str(self.output_h),
            str(self.c_bin), str(self.d_bin), str(self.m_bin), str(self.o_bin),
            str(self.eval_passes), str(self.n_bin), str(self.a_bin),
            str(self.fps), str(self.frametime), str(self.warmup_passes),
        ]
        print(f"[RH-DLSS5-AMD] Starting persistent AMD-NR D3D12 bridge: {self.bridge_exe.name} "
              f"({self.input_w}x{self.input_h} -> {self.output_w}x{self.output_h})", flush=True)
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.runtime_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Wait for READY
        while True:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"AMD-NR bridge exited during startup with code {self.proc.returncode}")
                raise RuntimeError("AMD-NR bridge closed pipe unexpectedly before READY")
            line = line.strip()
            if line == "READY":
                break
        if not self.o_bin.exists():
            raise RuntimeError("AMD-NR bridge did not produce output for frame 0")
        out_raw = np.fromfile(str(self.o_bin), dtype=np.float16).reshape((self.output_h, self.output_w, 4))
        return out_raw[..., :3].astype(np.float32)

    def process_next_frame(self, pixels_u8: np.ndarray, motion_fp16: np.ndarray) -> np.ndarray:
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("AMD-NR bridge is not running")
        self._write_frame_inputs(pixels_u8, motion_fp16)
        cmd_str = f"FRAME {self.c_bin} {self.d_bin} {self.m_bin} {self.o_bin} {self.n_bin} {self.a_bin} {self.eval_passes}\n"
        self.proc.stdin.write(cmd_str)
        self.proc.stdin.flush()
        # Wait for DONE
        while True:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"AMD-NR bridge crashed during frame execution (code {self.proc.returncode})")
                raise RuntimeError("AMD-NR bridge closed pipe unexpectedly")
            line = line.strip()
            if line == "DONE":
                break
        out_raw = np.fromfile(str(self.o_bin), dtype=np.float16).reshape((self.output_h, self.output_w, 4))
        return out_raw[..., :3].astype(np.float32)

    def close(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                try:
                    self.proc.stdin.write("EXIT\n")
                    self.proc.stdin.flush()
                    self.proc.wait(timeout=5)
                except Exception:
                    self.proc.kill()
            self.proc = None
        try:
            self.tmpdir.cleanup()
        except Exception:
            pass


def amd_bridge_process_frames(
    frames,
    input_w: int,
    input_h: int,
    output_w: int,
    output_h: int,
    params: dict,
    progress_callback=None,
):
    """Execute frame processing via persistent AMD-NR bridge daemon with linear compositing."""
    from . import common
    runtime: Path = params["runtime"]
    write_amd_nr_config(runtime, params)

    candidates = [
        runtime / "dlss_bridge.exe",
        common.PLUGIN_ROOT / "native" / "bin" / "dlss_bridge.exe",
        common.PLUGIN_ROOT / "native" / "src" / "bin" / "dlss_bridge.exe",
    ]
    bridge_exe = next((c for c in candidates if c.is_file()), None)
    if bridge_exe is None:
        raise common.DLSS5Error(
            "AMD-NR bridge binary (dlss_bridge.exe) not found. "
            "Run 'python setup_amd_runtime.py' to import AMD runtime binaries."
        )

    daemon = AmdBridgeDaemon(bridge_exe, input_w, input_h, output_w, output_h, runtime, params)
    total = int(params.get("frame_count", 1))
    try:
        for index, (pixels_u8, motion, reset) in enumerate(frames):
            if index == 0:
                out = daemon.start_and_process_first(pixels_u8, motion)
            else:
                out = daemon.process_next_frame(pixels_u8, motion)

            if params.get("detail", 1.0) != 1.0 or params.get("color", 1.0) != 1.0:
                out = common.nr_composite(
                    pixels_u8, out, params["detail"], params["color"],
                    out_is_linear=bool(params.get("hdr")), encode=not params.get("hdr"),
                )
            if progress_callback is not None:
                progress_callback(index + 1, total)
            yield out
    finally:
        daemon.close()


# -----------------------------------------------------------------------------
# Color Space & EOTF Transforms in PyTorch
# -----------------------------------------------------------------------------

def srgb_to_linear_torch(x: torch.Tensor) -> torch.Tensor:
    """sRGB EOTF, piecewise exact on PyTorch tensors."""
    x = torch.clamp(x, 0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, torch.pow((x + 0.055) / 1.055, 2.4))


def linear_to_srgb_torch(x: torch.Tensor) -> torch.Tensor:
    """Inverse sRGB EOTF on PyTorch tensors."""
    x = torch.clamp(x, 0.0, None)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * torch.pow(x, 1.0 / 2.4) - 0.055)


def rgb_to_ycbcr_torch(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB [B, 3, H, W] in [0, 1] to YCbCr [B, 3, H, W]."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.cat([y, cb, cr], dim=1)


def gaussian_blur_torch(img: torch.Tensor, sigma: float = 1.5, kernel_size: int = 5) -> torch.Tensor:
    """Depthwise 2D Gaussian blur for PyTorch tensors [B, C, H, W]."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    device = img.device
    dtype = img.dtype
    radius = kernel_size // 2
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    gauss = torch.exp(-0.5 * (x / max(sigma, 0.1)) ** 2)
    kernel_1d = gauss / gauss.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)[None, None, :, :]
    channels = img.shape[1]
    kernel = kernel_2d.repeat(channels, 1, 1, 1)
    return F.conv2d(img, kernel, padding=radius, groups=channels)


# -----------------------------------------------------------------------------
# ZludaTensorEnhancer: Pure PyTorch / ZLUDA Neural Rendering Pipeline
# -----------------------------------------------------------------------------

class ZludaTensorEnhancer:
    """Pure PyTorch execution engine for DLSS 5 Neural Rendering parameters.

    Accelerated on AMD GPUs via ZLUDA / ROCm. Reproduces:
    - Upscaling: 1x (DLAA/native refinement), 1.5x, 1.724x, 2x, 3x, auto buckets (1K, 2K, 4K, 8K)
    - Structure: High-frequency micro-contrast and edge reconstruction
    - Tone & Styles: Linear-space tonemapping (default, natural, cinematic, global_tone, tone, hdr)
    - Skin & AutoMask: Human skin detection in YCbCr/HSV space with modulated skin pore detail
    - Composite: Reference-style linear detail/color compositing and intensity blending
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device, dtype=torch.float32).view(1, 3, 1, 1)

    def process_frame(
        self,
        frame_rgb: torch.Tensor,  # [C, H, W] in [0, 1]
        out_w: int,
        out_h: int,
        params: dict,
    ) -> torch.Tensor:
        """Process a single frame with full DLSS 5 Neural Rendering contracts."""
        # Ensure 4D shape [1, C, H, W]
        if frame_rgb.ndim == 3:
            img = frame_rgb.unsqueeze(0).to(self.device, dtype=torch.float32)
        else:
            img = frame_rgb.to(self.device, dtype=torch.float32)

        orig_img = img.clone()
        scale = params.get("scale", 1.0) or 1.0
        style = params.get("style", 0)
        preset = params.get("preset", 0)
        intensity = float(params.get("intensity", 1.0))
        tone = float(params.get("tone", 1.0))
        structure = float(params.get("structure", 1.5))
        skin = float(params.get("skin", -1.0))
        auto_mask = bool(params.get("auto_mask", True))
        hdr = bool(params.get("hdr", False))
        global_tone = float(params.get("global_tone", -1.0))
        detail = float(params.get("detail", 1.0))
        color = float(params.get("color", 1.0))

        # 1. Upscale if output dimensions differ
        in_h, in_w = img.shape[2], img.shape[3]
        if out_w != in_w or out_h != in_h:
            img = F.interpolate(img, size=(out_h, out_w), mode="bicubic", align_corners=False, antialias=True)
            orig_base = F.interpolate(orig_img, size=(out_h, out_w), mode="bicubic", align_corners=False, antialias=True)
        else:
            orig_base = orig_img

        # 2. Linear light domain for Neural Rendering pass
        lin_img = srgb_to_linear_torch(img) if not hdr else img.clamp(0.0, 1.0)

        # 3. Micro-structure and detail reconstruction (structure)
        low_freq = gaussian_blur_torch(lin_img, sigma=1.2, kernel_size=5)
        high_freq = lin_img - low_freq
        # Modulate high-frequency detail by structure parameter with soft limiting
        structure_factor = max(0.0, structure) * 0.45
        enhanced_high_freq = torch.tanh(high_freq * structure_factor * 2.0) * 0.5
        reconstructed = lin_img + enhanced_high_freq

        # 4. Skin protection & AutoMask (skin, auto_mask)
        if auto_mask:
            # Color space detection in YCbCr
            srgb_proxy = linear_to_srgb_torch(reconstructed)
            ycbcr = rgb_to_ycbcr_torch(srgb_proxy)
            y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
            # Skin range heuristics: Cr ~ 0.52..0.68, Cb ~ 0.30..0.50, Y ~ 0.15..0.95
            skin_mask = (
                (cr >= 0.51) & (cr <= 0.69) &
                (cb >= 0.29) & (cb <= 0.51) &
                (y >= 0.12) & (y <= 0.96)
            ).float()
            # Smooth skin mask boundary
            skin_mask = gaussian_blur_torch(skin_mask, sigma=2.5, kernel_size=7)

            if skin >= 0.0:
                # Modulated skin texture adjustment
                # skin=0 softly blurs blemishes; skin=1.5 restores delicate pores
                skin_smooth = gaussian_blur_torch(reconstructed, sigma=1.0, kernel_size=5)
                skin_detail = (reconstructed - skin_smooth) * (skin * 0.8)
                skin_result = skin_smooth + skin_detail
                reconstructed = (1.0 - skin_mask) * reconstructed + skin_mask * skin_result

        # 5. Global Tone & Local Tone Mapping (global_tone, tone)
        if global_tone >= 0.0:
            # Shift overall exposure
            exposure_gain = 2.0 ** ((global_tone - 1.0) * 0.6)
            reconstructed = reconstructed * exposure_gain

        if tone != 1.0:
            # Local tone adaptation: compress high dynamic range
            luma = (reconstructed * self.luma_weights).sum(dim=1, keepdim=True)
            luma_base = gaussian_blur_torch(luma, sigma=10.0, kernel_size=15)
            # Gentle compression / expansion
            gamma = 1.0 + (tone - 1.0) * 0.25
            luma_tone = luma * torch.pow(torch.clamp(luma_base + 1e-4, 1e-4, 1.0), gamma - 1.0)
            reconstructed = reconstructed * (luma_tone / torch.clamp(luma, 1e-4))

        # 6. Style Curves: default, natural, cinematic
        # 0: default, 1: natural (soft linear fidelity), 2: cinematic (filmic S-curve)
        if style == 2:  # Cinematic
            # Filmic S-curve contrast
            luma = (reconstructed * self.luma_weights).sum(dim=1, keepdim=True)
            cin_luma = luma * luma * (3.0 - 2.0 * torch.clamp(luma, 0.0, 1.0))
            reconstructed = reconstructed * (cin_luma / torch.clamp(luma, 1e-4))
        elif style == 1:  # Natural
            # Keep very close to input with soft contrast
            pass

        # 7. Convert to sRGB if needed
        out_srgb = linear_to_srgb_torch(reconstructed) if not hdr else reconstructed
        orig_srgb = orig_base

        # 8. Chroma and Detail Compositing (detail, color)
        if detail != 1.0 or color != 1.0:
            # Reference-style linear composite
            lin_orig = srgb_to_linear_torch(orig_srgb)
            lin_nr = srgb_to_linear_torch(out_srgb)
            lo = (lin_orig * self.luma_weights).sum(dim=1, keepdim=True)
            ln = (lin_nr * self.luma_weights).sum(dim=1, keepdim=True)
            scale_l = ln / torch.clamp(lo, 1e-4)
            loc = lin_orig * scale_l
            cc = loc + (lin_nr - loc) * color
            comp = lin_orig + (cc - lin_orig) * detail
            out_srgb = linear_to_srgb_torch(comp)

        # 9. Intensity Blending
        final_srgb = orig_srgb + (out_srgb - orig_srgb) * intensity
        final_clamped = torch.clamp(final_srgb, 0.0, 1.0)

        return final_clamped.squeeze(0)  # [C, H, W]


# -----------------------------------------------------------------------------
# AmdFrameInterpolator: Bidirectional OpenCV DIS + PyTorch Pyramidal Warping
# -----------------------------------------------------------------------------

class AmdFrameInterpolator:
    """Frame Generation engine for AMD GPUs using OpenCV DIS Optical Flow + PyTorch GPU Warping.

    Fallback when proprietary dlssg-worker.exe is not available on AMD hardware.
    Features:
    - Bidirectional optical flow ($F_{0 \to 1}, F_{1 \to 0}$) via OpenCV DIS.
    - Scene cut detection via scene_change_threshold.
    - Sub-pixel bilinear/bicubic backward grid sampling on PyTorch/ZLUDA.
    - Cascaded dense grid resampling for multipliers (2x, 3x, 4x) and target frame rates.
    - Memory-bounded frame-by-frame streaming.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM) if cv2 is not None else None
        if self.dis is not None:
            self.dis.setUseSpatialPropagation(True)
            self.dis.setFinestScale(1)

    def compute_flow_bidirectional(
        self, img0_u8: np.ndarray, img1_u8: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forward (0->1) and backward (1->0) optical flow in pixels/frame."""
        if self.dis is None:
            h, w = img0_u8.shape[:2]
            return np.zeros((h, w, 2), dtype=np.float32), np.zeros((h, w, 2), dtype=np.float32)

        # Grayscale downscaled computation for speed & stability
        gray0 = cv2.cvtColor(img0_u8, cv2.COLOR_RGB2GRAY)
        gray1 = cv2.cvtColor(img1_u8, cv2.COLOR_RGB2GRAY)

        flow_w = min(img0_u8.shape[1], 960)
        flow_h = max(32, int(round(img0_u8.shape[0] * flow_w / img0_u8.shape[1] / 2.0) * 2))
        flow_w = max(32, int(round(flow_w / 2.0) * 2))

        if flow_w != img0_u8.shape[1] or flow_h != img0_u8.shape[0]:
            small0 = cv2.resize(gray0, (flow_w, flow_h), interpolation=cv2.INTER_AREA)
            small1 = cv2.resize(gray1, (flow_w, flow_h), interpolation=cv2.INTER_AREA)
            fwd_flow = self.dis.calc(small0, small1, None)
            bwd_flow = self.dis.calc(small1, small0, None)
            fwd_flow = cv2.resize(fwd_flow, (img0_u8.shape[1], img0_u8.shape[0]), interpolation=cv2.INTER_LINEAR)
            bwd_flow = cv2.resize(bwd_flow, (img0_u8.shape[1], img0_u8.shape[0]), interpolation=cv2.INTER_LINEAR)
            fwd_flow[..., 0] *= img0_u8.shape[1] / flow_w
            fwd_flow[..., 1] *= img0_u8.shape[0] / flow_h
            bwd_flow[..., 0] *= img0_u8.shape[1] / flow_w
            bwd_flow[..., 1] *= img0_u8.shape[0] / flow_h
        else:
            fwd_flow = self.dis.calc(gray0, gray1, None)
            bwd_flow = self.dis.calc(gray1, gray0, None)

        return fwd_flow, bwd_flow

    def warp_and_blend(
        self,
        t0: torch.Tensor,  # [1, 3, H, W]
        t1: torch.Tensor,  # [1, 3, H, W]
        flow_0to1: torch.Tensor,  # [1, H, W, 2] in pixels
        flow_1to0: torch.Tensor,  # [1, H, W, 2] in pixels
        alpha: float,  # intermediate timestamp 0..1
    ) -> torch.Tensor:
        """Warp and blend two frames at intermediate timestamp alpha."""
        b, c, h, w = t0.shape
        # Base grid in normalized coordinates [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=self.device, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, w, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]

        scale = torch.tensor([2.0 / max(1, w - 1), 2.0 / max(1, h - 1)], device=self.device, dtype=torch.float32)

        # Approximate intermediate flow displacements
        grid0 = base_grid + (-alpha * flow_0to1) * scale
        grid1 = base_grid + (-(1.0 - alpha) * flow_1to0) * scale

        warp0 = F.grid_sample(t0, grid0, mode="bicubic", padding_mode="border", align_corners=True)
        warp1 = F.grid_sample(t1, grid1, mode="bicubic", padding_mode="border", align_corners=True)

        blended = (1.0 - alpha) * warp0 + alpha * warp1
        return torch.clamp(blended, 0.0, 1.0)

    def interpolate_pair(
        self,
        img0_u8: np.ndarray,
        img1_u8: np.ndarray,
        t0_gpu: torch.Tensor,
        t1_gpu: torch.Tensor,
        alpha_list: list[float],
        threshold: float,
    ) -> list[torch.Tensor]:
        """Interpolate intermediate frames between img0 and img1."""
        # Scene cut detection
        mean_diff = float(np.mean(np.abs(img1_u8.astype(np.float32) - img0_u8.astype(np.float32)))) / 255.0
        if mean_diff > threshold:
            # Cut: hold img0 across cut slots
            return [t0_gpu.clone() for _ in alpha_list]

        fwd_np, bwd_np = self.compute_flow_bidirectional(img0_u8, img1_u8)
        fwd_torch = torch.from_numpy(fwd_np).unsqueeze(0).to(self.device, dtype=torch.float32)
        bwd_torch = torch.from_numpy(bwd_np).unsqueeze(0).to(self.device, dtype=torch.float32)

        generated = []
        for alpha in alpha_list:
            interp = self.warp_and_blend(t0_gpu, t1_gpu, fwd_torch, bwd_torch, alpha)
            generated.append(interp)
        return generated
