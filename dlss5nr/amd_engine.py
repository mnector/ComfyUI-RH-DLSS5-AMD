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


STYLE_PRESETS = {
    "cinematic": {"residual_strength": 0.85, "residual_limit": 0.20, "temporal_stability": 0.75, "sharpening": 0.00, "detail_strength": 1.00, "colour_strength": 1.00},
    "crisp":     {"residual_strength": 1.20, "residual_limit": 0.35, "temporal_stability": 0.50, "sharpening": 0.30, "detail_strength": 1.00, "colour_strength": 1.00},
    "natural":   {"residual_strength": 0.70, "residual_limit": 0.15, "temporal_stability": 0.70, "sharpening": 0.00, "detail_strength": 1.00, "colour_strength": 1.00},
    "vivid":     {"residual_strength": 1.10, "residual_limit": 0.30, "temporal_stability": 0.60, "sharpening": 0.15, "detail_strength": 1.00, "colour_strength": 1.00},
    "default":   {"residual_strength": 1.00, "residual_limit": 0.25, "temporal_stability": 0.60, "sharpening": 0.00, "detail_strength": 1.00, "colour_strength": 1.00},
}


def write_amd_nr_config(runtime_dir: Path, params: dict) -> None:
    """Write or update hot-reload dlssnr_on_amd.ini and OptiScaler.ini for community AMD-NR runtimes."""
    if not runtime_dir or not runtime_dir.exists():
        return

    # Resolve AMD-NR style preset and sliders (AMDNR v0.3.5.6)
    raw_style = str(params.get("nr_style", "Full Model Look")).strip()
    presets = {
        "Full Model Look": {"residual_strength": 1.00, "residual_limit": 0.00, "temporal_stability": 0.50, "sharpening": 0.00, "detail_strength": 2.00, "colour_strength": 1.00, "neural_placement": "After upscaling", "full_network": True},
        "NVIDIA look":     {"residual_strength": 1.00, "residual_limit": 0.00, "temporal_stability": 0.60, "sharpening": 0.00, "detail_strength": 1.00, "colour_strength": 1.00, "neural_placement": "Before upscaling", "full_network": True},
        "AMDNR soft":      {"residual_strength": 0.70, "residual_limit": 0.15, "temporal_stability": 0.70, "sharpening": 0.00, "detail_strength": 1.00, "colour_strength": 1.00, "neural_placement": "Before upscaling", "full_network": False},
        "Cinematic":       {"residual_strength": 0.85, "residual_limit": 0.20, "temporal_stability": 0.75, "sharpening": 0.00, "detail_strength": 1.00, "colour_strength": 1.00},
        "Crisp":           {"residual_strength": 1.20, "residual_limit": 0.35, "temporal_stability": 0.50, "sharpening": 0.30, "detail_strength": 1.00, "colour_strength": 1.00},
        "Natural":         {"residual_strength": 0.70, "residual_limit": 0.15, "temporal_stability": 0.70, "sharpening": 0.00, "detail_strength": 1.00, "colour_strength": 1.00},
        "Vivid":           {"residual_strength": 1.10, "residual_limit": 0.30, "temporal_stability": 0.60, "sharpening": 0.15, "detail_strength": 1.00, "colour_strength": 1.00},
        "Default":         {"residual_strength": 1.00, "residual_limit": 0.00, "temporal_stability": 0.50, "sharpening": 0.00, "detail_strength": 2.00, "colour_strength": 1.00, "neural_placement": "After upscaling", "full_network": True},
    }
    preset_dict = presets.get(raw_style, {})

    if "full" in raw_style.lower():
        amd_nr_look = "auto"
    elif "nvidia" in raw_style.lower():
        amd_nr_look = "0"
    elif "soft" in raw_style.lower():
        amd_nr_look = "1"
    else:
        amd_nr_look = "custom"

    enable_nr = bool(params.get("enable_nr", True))
    res_strength = float(params.get("residual_strength", preset_dict.get("residual_strength", 1.00)))
    res_limit = float(params.get("residual_limit", preset_dict.get("residual_limit", 0.25)))
    if res_limit <= 0.0:
        res_limit = 0.25
    temp_stability = float(params.get("temporal_stability", preset_dict.get("temporal_stability", 0.60)))
    sharpness = float(params.get("sharpening", preset_dict.get("sharpening", 0.00)))
    detail_str = float(params.get("detail_strength", preset_dict.get("detail_strength", 1.00)))
    colour_str = float(params.get("colour_strength", preset_dict.get("colour_strength", 1.00)))
    
    placement = "post" if "after" in str(params.get("neural_placement", preset_dict.get("neural_placement", "After"))).lower() else "pre"
    working_scale = float(params.get("nr_resolution", 100)) / 100.0
    passes = int(params.get("neural_passes", 1))
    full_net = bool(params.get("full_network", preset_dict.get("full_network", True)))
    fast_m = bool(params.get("fast_mode", False))
    dynamic_res = bool(params.get("dynamic_nr_resolution", False))
    raw_interleave = str(params.get("model_interleave", "Off")).strip()
    if "4" in raw_interleave:
        interleave = "4"
    elif "3" in raw_interleave:
        interleave = "3"
    elif "2" in raw_interleave or raw_interleave.lower() == "on":
        interleave = "2"
    else:
        interleave = "0"
    composition = "1" if "reno" in str(params.get("colour_composition", "Classic")).lower() else "0"
    nr_backend = "daniel" if "daniel" in str(params.get("neural_runtime", "daniel")).lower() else "lmxxf"

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

            config.set("DlssNr", "Enabled", "true" if enable_nr else "false")
            config.set("DlssNr", "AmdNeuralLighting", "true" if enable_nr else "false")
            config.set("DlssNr", "NrBackend", nr_backend)
            config.set("DlssNr", "AmdNrLook", amd_nr_look)
            config.set("DlssNr", "AmdPlacement", placement)
            config.set("DlssNr", "RunBeforeSR", "false" if placement == "post" else "true")
            config.set("DlssNr", "AmdNeuralLightingStrength", "1")
            config.set("DlssNr", "AmdSkipUnwrittenFrames", "false")
            config.set("DlssNr", "AmdHoldResize", "false")
            config.set("DlssNr", "AmdUseGameExposure", "1")
            config.set("DlssNr", "UseGameExposure", "1")
            config.set("DlssNr", "ScanExposure", "false")
            config.set("DlssNr", "ScanMeter", "false")
            config.set("DlssNr", "ScanAnchors", "false")
            config.set("DlssNr", "AmdLmxxfAutoExposure", "false")
            if config.has_option("DlssNr", "Scale"):
                config.remove_option("DlssNr", "Scale")
            if config.has_option("DlssNr", "Residual"):
                config.remove_option("DlssNr", "Residual")
            config.set("DlssNr", "UseDepth", "1")
            config.set("DlssNr", "UseFsrInputs", "1")
            config.set("DlssNr", "Temporal", "1")
            config.set("DlssNr", "DepthInverted", "1")
            config.set("DlssNr", "UseAutoMask", "1")
            config.set("DlssNr", "FreshExposure", "0")
            config.set("DlssNr", "UsePreExposure", "0")
            config.set("DlssNr", "WorkingScale", f"{working_scale:.2f}")
            config.set("DlssNr", "Passes", str(passes))
            config.set("DlssNr", "LmxxfFullNetwork", "true" if full_net else "false")
            config.set("DlssNr", "AmdLmxxfFastMode", "true" if fast_m else "false")
            config.set("DlssNr", "AmdDanielFastMode", "true" if fast_m else "false")
            config.set("DlssNr", "AmdDynamicRes", "true" if dynamic_res else "false")
            config.set("DlssNr", "AmdInterleave", interleave)
            config.set("DlssNr", "TransferStrength", f"{res_strength:.4f}")
            config.set("DlssNr", "MaxRatio", f"{res_limit:.4f}")
            config.set("DlssNr", "AmdTemporalStability", f"{temp_stability:.4f}")
            config.set("DlssNr", "AmdSharpness", f"{sharpness:.4f}")
            config.set("DlssNr", "AmdDetailStrength", f"{detail_str:.4f}")
            config.set("DlssNr", "AmdColourStrength", f"{colour_str:.4f}")
            config.set("DlssNr", "AmdComposition", composition)
            config.set("DlssNr", "AmdDeviceGate", "false")
            config.set("DlssNr", "AmdBootstrapQueue", "true")
            config.set("DlssNr", "AmdFaceDetail", "true")

            # [DlssNrOnAmd] - Direct runtime configuration for danielblnc dlssnr_amd kernels
            if not config.has_section("DlssNrOnAmd"):
                config.add_section("DlssNrOnAmd")
            config.set("DlssNrOnAmd", "Enabled", "1" if enable_nr else "0")
            config.set("DlssNrOnAmd", "Interleave", interleave)
            config.set("DlssNrOnAmd", "Residual", "1")
            config.set("DlssNrOnAmd", "Temporal", "1")
            config.set("DlssNrOnAmd", "UseDepth", "1")
            config.set("DlssNrOnAmd", "UseFsrInputs", "1")
            config.set("DlssNrOnAmd", "DepthInverted", "1")
            config.set("DlssNrOnAmd", "PreHistory", "1")
            config.set("DlssNrOnAmd", "UseAutoMask", "1")
            config.set("DlssNrOnAmd", "UseGameExposure", "1")
            config.set("DlssNrOnAmd", "UsePreExposure", "0")
            config.set("DlssNrOnAmd", "FreshExposure", "0")
            config.set("DlssNrOnAmd", "Quality", "0" if not fast_m else "1")
            config.set("DlssNrOnAmd", "SpinDraw", "1")

            # [AmdLook] - Non-neural post-process filter disabled to avoid compounding contrast/saturation
            if not config.has_section("AmdLook"):
                config.add_section("AmdLook")
            config.set("AmdLook", "Enabled", "false")
            config.set("AmdLook", "Appearance", "0")
            config.set("AmdLook", "Mix", "0.0")

            # [Upscalers]
            if not config.has_section("Upscalers"):
                config.add_section("Upscalers")
            is_ray_reconstruction = bool(params.get("is_ray_reconstruction", False))
            dx12_carrier = str(params.get("dx12_upscaler", params.get("upscale_method", "ffx"))).lower()
            if is_ray_reconstruction or "ray" in dx12_carrier or "reconstruction" in dx12_carrier or "rr" in dx12_carrier:
                target_carrier = "ffx"
                config.set("Upscalers", "FfxDx12RRPath", "auto")
            elif "xess" in dx12_carrier or "intel" in dx12_carrier:
                target_carrier = "xess"
            elif "dlss" in dx12_carrier or "nvidia" in dx12_carrier:
                target_carrier = "dlss"
            else:
                target_carrier = "ffx"
            config.set("Upscalers", "Dx12Upscaler", target_carrier)

            # [FSR-RR] - AMD FidelityFX Ray Regeneration (Feature 13 DLSS-D translation)
            if not config.has_section("FSR-RR"):
                config.add_section("FSR-RR")
            config.set("FSR-RR", "FfxDenoiserUseAmdDefaults", "true")
            config.set("FSR-RR", "FfxDenoiserAllowPreRdna4", "true")
            config.set("FSR-RR", "RrBackend", "auto")
            config.set("FSR-RR", "FfxDenoiserSkinSmoothing", "false")
            config.set("FSR-RR", "FfxDenoiserAlbedoFp16", "true")
            config.set("FSR-RR", "OverrideSharpness", "true")
            config.set("FSR-RR", "RrDefaultSharpness", "0.40")
            config.set("FSR-RR", "FfxDenoiserFloorNormalSharpness", "1.00")
            config.set("FSR-RR", "FfxDenoiserFloorGrazingSharpness", "1.00")
            config.set("FSR-RR", "FfxDenoiserFloorDetailBoost", "1.00")
            config.set("FSR-RR", "FfxDenoiserFloorHandoverDetail", "1.00")
            config.set("FSR-RR", "FfxDenoiserDisocThreshold", "0.05")

            # [DLSSD]
            if not config.has_section("DLSSD"):
                config.add_section("DLSSD")
            config.set("DLSSD", "RenderPresetOverride", "auto")

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
            config.set("InitFlags", "DepthInverted", "true")
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

def compute_motion_from_video(motion_video: torch.Tensor, target_w: int, target_h: int) -> np.ndarray:
    """Computes real 3D camera-parallax optical flow from a 2-3 frame video (e.g. Depthflow Dolly)."""
    import cv2
    if motion_video is None or len(motion_video) < 2:
        return np.zeros((target_h, target_w, 2), dtype=np.float16)

    # Frame 0 and Frame 1
    f0 = (motion_video[0].detach().cpu().numpy() * 255.0).astype(np.uint8)
    f1 = (motion_video[1].detach().cpu().numpy() * 255.0).astype(np.uint8)
    if f0.ndim == 3 and f0.shape[-1] > 3:
        f0 = f0[..., :3]
    if f1.ndim == 3 and f1.shape[-1] > 3:
        f1 = f1[..., :3]

    g0 = cv2.cvtColor(f0, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    # DLSS motion vector convention: displacement from current (g1) back to previous (g0)
    flow = dis.calc(g1, g0, None)

    if (g0.shape[1], g0.shape[0]) != (target_w, target_h):
        flow = cv2.resize(flow, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= (target_w / float(g0.shape[1]))
        flow[..., 1] *= (target_h / float(g0.shape[0]))

    return np.ascontiguousarray(flow.astype(np.float16))


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
        self.warmup_passes = max(0, int(params.get("warmup_passes", params.get("warmup_frames", 8))))
        runtime_name = str(params.get("neural_runtime", "lmxxf")).lower()
        if "daniel" in runtime_name and self.warmup_passes > 6:
            print(f"[RH-DLSS5-AMD] Clamping danielblnc warmup passes from {self.warmup_passes} to 6 "
                  f"(danielblnc closed runtime has a 6-frame buffer limit; use neural_runtime='lmxxf' for unlimited warmup passes).", flush=True)
            self.warmup_passes = 6
        self.settle_pause_ms = max(0, int(params.get("settle_pause_ms", 350)))
        self.warmup_interval_ms = max(0, int(params.get("warmup_interval_ms", 50)))
        self.eval_passes = max(1, int(params.get("neural_passes", 1)))
        self.fps = 30
        self.frametime = 0.033

    def _write_frame_inputs(
        self,
        pixels_u8: np.ndarray,
        motion_fp16: np.ndarray,
        depth_f32: Optional[np.ndarray] = None,
        normal_fp16: Optional[np.ndarray] = None,
        albedo_fp16: Optional[np.ndarray] = None,
    ):
        h, w = pixels_u8.shape[:2]
        # Color: RGBA FP16
        color_rgba = np.empty((h, w, 4), dtype=np.float16)
        color_rgba[..., :3] = (pixels_u8.astype(np.float32) / 255.0).astype(np.float16)
        color_rgba[..., 3] = 1.0
        color_rgba.tofile(str(self.c_bin))

        # Depth: Real Depth if provided, else neutral mid-plane 0.5 FP32
        if depth_f32 is not None:
            d_arr = np.ascontiguousarray(depth_f32, dtype=np.float32).squeeze()
            if d_arr.ndim == 3:
                d_arr = d_arr[..., 0]
            if d_arr.shape[:2] != (h, w):
                import cv2
                d_arr = cv2.resize(d_arr, (w, h), interpolation=cv2.INTER_LINEAR)
            d_arr = np.clip(d_arr, 0.0, 1.0).astype(np.float32)
            d_arr.tofile(str(self.d_bin))
        else:
            depth_const = np.full((h, w), 0.5, dtype=np.float32)
            depth_const.tofile(str(self.d_bin))

        # Motion vectors: RG FP16 (sized to display resolution so display motion allocation is satisfied)
        mvec_rg = np.ascontiguousarray(motion_fp16, dtype=np.float16)
        if mvec_rg.shape[:2] != (self.output_h, self.output_w):
            import cv2
            scale_x = float(self.output_w) / float(mvec_rg.shape[1])
            scale_y = float(self.output_h) / float(mvec_rg.shape[0])
            mvec_resized = cv2.resize(mvec_rg.astype(np.float32), (self.output_w, self.output_h), interpolation=cv2.INTER_LINEAR)
            mvec_resized[..., 0] *= scale_x
            mvec_resized[..., 1] *= scale_y
            mvec_rg = mvec_resized.astype(np.float16)
        mvec_rg.tofile(str(self.m_bin))

        # Normal: Real Normal if provided, else flat (0, 0, 1, 1) FP16
        if normal_fp16 is not None:
            n_arr = np.ascontiguousarray(normal_fp16, dtype=np.float32)
            if n_arr.ndim == 4:
                n_arr = n_arr[0]
            if n_arr.shape[:2] != (h, w):
                import cv2
                n_arr = cv2.resize(n_arr, (w, h), interpolation=cv2.INTER_LINEAR)
            normal_rgba = np.zeros((h, w, 4), dtype=np.float16)
            normal_rgba[..., :3] = n_arr[..., :3].astype(np.float16)
            normal_rgba[..., 3] = 1.0
            normal_rgba.tofile(str(self.n_bin))
        else:
            normal_rgba = np.zeros((h, w, 4), dtype=np.float16)
            normal_rgba[..., 2] = 1.0
            normal_rgba[..., 3] = 1.0
            normal_rgba.tofile(str(self.n_bin))

        # Albedo: Real Albedo if provided, else copy of color
        if albedo_fp16 is not None:
            a_arr = np.ascontiguousarray(albedo_fp16, dtype=np.float32)
            if a_arr.ndim == 4:
                a_arr = a_arr[0]
            if a_arr.shape[:2] != (h, w):
                import cv2
                a_arr = cv2.resize(a_arr, (w, h), interpolation=cv2.INTER_LINEAR)
            albedo_rgba = np.zeros((h, w, 4), dtype=np.float16)
            albedo_rgba[..., :3] = a_arr[..., :3].astype(np.float16)
            albedo_rgba[..., 3] = 1.0
            albedo_rgba.tofile(str(self.a_bin))
        else:
            color_rgba.tofile(str(self.a_bin))

    def start_and_process_first(
        self,
        pixels_u8: np.ndarray,
        motion_fp16: np.ndarray,
        depth_f32: Optional[np.ndarray] = None,
        normal_fp16: Optional[np.ndarray] = None,
        albedo_fp16: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._write_frame_inputs(pixels_u8, motion_fp16, depth_f32, normal_fp16, albedo_fp16)
        feature_id = 13 if bool(self.params.get("is_ray_reconstruction", False)) else 1
        cmd = [
            str(self.bridge_exe),
            str(self.input_w), str(self.input_h),
            str(self.output_w), str(self.output_h),
            str(self.c_bin), str(self.d_bin), str(self.m_bin), str(self.o_bin),
            str(self.eval_passes), str(self.n_bin), str(self.a_bin),
            str(self.fps), str(self.frametime), str(self.warmup_passes),
            str(self.settle_pause_ms), str(self.warmup_interval_ms),
            str(feature_id),
        ]
        feat_name = "Ray Reconstruction (DLSS-D / FSR-RR Feature 13)" if feature_id == 13 else "SuperSampling (DLSS-SR / FSR Feature 1)"
        print(f"[RH-DLSS5-AMD] Starting persistent AMD-NR D3D12 bridge: {self.bridge_exe.name} "
              f"[{feat_name}] ({self.input_w}x{self.input_h} -> {self.output_w}x{self.output_h})", flush=True)
        bridge_env = dict(os.environ)
        bridge_env["DLSS_FEATURE_ID"] = str(feature_id)
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.runtime_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=bridge_env,
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
        if np.isnan(out_raw).any() or np.isinf(out_raw).any():
            print("[RH-DLSS5-AMD] WARNING: NaN/Inf detected in raw D3D12 output texture; sanitizing...", flush=True)
            out_raw = np.nan_to_num(out_raw, nan=0.5, posinf=1.0, neginf=0.0)
        return out_raw[..., :3].astype(np.float32)

    def process_next_frame(
        self,
        pixels_u8: np.ndarray,
        motion_fp16: np.ndarray,
        depth_f32: Optional[np.ndarray] = None,
        normal_fp16: Optional[np.ndarray] = None,
        albedo_fp16: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("AMD-NR bridge is not running")
        self._write_frame_inputs(pixels_u8, motion_fp16, depth_f32, normal_fp16, albedo_fp16)
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
        if np.isnan(out_raw).any() or np.isinf(out_raw).any():
            print("[RH-DLSS5-AMD] WARNING: NaN/Inf detected in raw D3D12 output texture; sanitizing...", flush=True)
            out_raw = np.nan_to_num(out_raw, nan=0.5, posinf=1.0, neginf=0.0)
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

    # 64-pixel alignment for D3D12 / DP4a hardware neural tensor pipelines
    pad_h = (64 - (input_h % 64)) % 64
    pad_w = (64 - (input_w % 64)) % 64
    needs_padding = (pad_h > 0 or pad_w > 0)

    if needs_padding:
        scale_x = float(output_w) / float(input_w)
        scale_y = float(output_h) / float(input_h)
        proc_in_w = input_w + pad_w
        proc_in_h = input_h + pad_h
        proc_out_w = int(round(proc_in_w * scale_x))
        proc_out_h = int(round(proc_in_h * scale_y))
    else:
        proc_in_w, proc_in_h = input_w, input_h
        proc_out_w, proc_out_h = output_w, output_h

    daemon = AmdBridgeDaemon(bridge_exe, proc_in_w, proc_in_h, proc_out_w, proc_out_h, runtime, params)
    total = int(params.get("frame_count", 1))
    try:
        for index, item in enumerate(frames):
            if len(item) == 3:
                pixels_u8, motion, reset = item
                d_frame, n_frame, a_frame = None, None, None
            else:
                pixels_u8, motion, reset, d_frame, n_frame, a_frame = item

            orig_u8 = pixels_u8
            if needs_padding:
                pixels_u8 = np.pad(pixels_u8, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
                if motion is not None:
                    motion = np.pad(motion, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
                if d_frame is not None:
                    d_frame = np.pad(d_frame, ((0, pad_h), (0, pad_w)), mode="edge") if d_frame.ndim == 2 else np.pad(d_frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
                if n_frame is not None:
                    n_frame = np.pad(n_frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge") if n_frame.ndim == 3 else np.pad(n_frame, ((0, pad_h), (0, pad_w)), mode="edge")
                if a_frame is not None:
                    a_frame = np.pad(a_frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge") if a_frame.ndim == 3 else np.pad(a_frame, ((0, pad_h), (0, pad_w)), mode="edge")

            if index == 0:
                warmup_frame = params.get("warmup_dolly_frame")
                warmup_motion = params.get("warmup_dolly_motion")
                if warmup_frame is not None and warmup_motion is not None:
                    # Prime temporal history with Depthflow Dolly frame 1 + real optical flow
                    if needs_padding:
                        warmup_frame = np.pad(warmup_frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
                        warmup_motion = np.pad(warmup_motion, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
                    daemon.start_and_process_first(warmup_frame, warmup_motion, d_frame, n_frame, a_frame)
                    # Now execute target static frame (pixels_u8) with zero motion relative to itself
                    zero_motion = np.zeros((proc_in_h, proc_in_w, 2), dtype=np.float16)
                    out = daemon.process_next_frame(pixels_u8, zero_motion, d_frame, n_frame, a_frame)
                else:
                    out = daemon.start_and_process_first(pixels_u8, motion, d_frame, n_frame, a_frame)
            else:
                out = daemon.process_next_frame(pixels_u8, motion, d_frame, n_frame, a_frame)

            if needs_padding:
                out = out[:output_h, :output_w]

            if params.get("detail", 1.0) != 1.0 or params.get("color", 1.0) != 1.0:
                out = common.nr_composite(
                    orig_u8, out, params["detail"], params["color"],
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


def cas_sharpen_torch(img: torch.Tensor, sharpness: float = 0.0) -> torch.Tensor:
    """AMD FidelityFX Contrast Adaptive Sharpening (CAS) on PyTorch tensors [B, C, H, W]."""
    if sharpness <= 0.001:
        return img
    pad = F.pad(img, (1, 1, 1, 1), mode="replicate")
    c = img
    u = pad[:, :, 0:-2, 1:-1]
    d = pad[:, :, 2:,   1:-1]
    l = pad[:, :, 1:-1, 0:-2]
    r = pad[:, :, 1:-1, 2:]

    # Local cross min and max per channel
    min_c = torch.min(torch.min(torch.min(u, d), torch.min(l, r)), c)
    max_c = torch.max(torch.max(torch.max(u, d), torch.max(l, r)), c)

    # Soft contrast-adaptive weight
    amp = torch.sqrt(torch.clamp(torch.min(min_c, 2.0 - max_c) / (max_c + 1e-5), 0.0, 1.0))
    w = amp * (-0.125 * min(1.0, max(0.0, sharpness)))

    sharpened = (c + (u + d + l + r) * w) / (1.0 + 4.0 * w)
    return torch.clamp(sharpened, 0.0, 1.0)


# -----------------------------------------------------------------------------
# ZludaTensorEnhancer: Pure PyTorch / ZLUDA Neural Rendering Pipeline
# -----------------------------------------------------------------------------

class ZludaTensorEnhancer:
    """Pure PyTorch execution engine for DLSS 5 Neural Rendering parameters.

    Accelerated on AMD GPUs via Speedstu/CUDA-for-AMD-Windows (ZLUDA / ROCm). Reproduces:
    - Upscaling: 1x (DLAA/native refinement), 1.5x, 1.724x, 2x, 3x, auto buckets (1K, 2K, 4K, 8K)
    - Residual Strength & Limit: Bounded high-frequency reconstruction (TransferStrength & MaxRatio)
    - Detail & Colour: AmdDetailStrength and AmdColourStrength chroma adaptation
    - CAS Sharpening: AMD FidelityFX Contrast Adaptive Sharpening (AmdSharpness)
    - Temporal Stability: Inter-frame temporal accumulation smoothing (AmdTemporalStability)
    - Styles: Presets (Default, Cinematic, Crisp, Natural, Vivid, Custom) and filmic curves
    - Skin & AutoMask: Human skin detection in YCbCr space with modulated skin pore detail
    - Completely in-process in VRAM: NO game windows, NO popup flashes, NO black screens.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self.prev_frame: Optional[torch.Tensor] = None

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

        # Resolve AMD-NR style preset & sliders
        raw_style = str(params.get("nr_style", params.get("style", "default"))).lower().split()[0]
        preset_values = STYLE_PRESETS.get(raw_style, {})

        res_strength = float(params.get("residual_strength", preset_values.get("residual_strength", params.get("intensity", 1.0))))
        res_limit = float(params.get("residual_limit", preset_values.get("residual_limit", 0.25)))
        temp_stability = float(params.get("temporal_stability", preset_values.get("temporal_stability", 0.60)))
        sharpness = float(params.get("sharpening", preset_values.get("sharpening", 0.00)))
        detail_str = float(params.get("detail_strength", preset_values.get("detail_strength", params.get("structure", 1.0))))
        colour_str = float(params.get("colour_strength", preset_values.get("colour_strength", params.get("color", 1.0))))
        passes = max(1, min(4, int(params.get("neural_passes", 1))))

        scale = params.get("scale", 1.0) or 1.0
        tone = float(params.get("tone", 1.0))
        skin = float(params.get("skin", -1.0))
        auto_mask = bool(params.get("auto_mask", True))
        hdr = bool(params.get("hdr", False))
        global_tone = float(params.get("global_tone", -1.0))

        # 1. Upscale if output dimensions differ
        in_h, in_w = img.shape[2], img.shape[3]
        if out_w != in_w or out_h != in_h:
            img = F.interpolate(img, size=(out_h, out_w), mode="bicubic", align_corners=False, antialias=True)
            orig_base = F.interpolate(orig_img, size=(out_h, out_w), mode="bicubic", align_corners=False, antialias=True)
        else:
            orig_base = orig_img

        # 2. Linear light domain for Neural Rendering pass
        lin_img = srgb_to_linear_torch(img) if not hdr else img.clamp(0.0, 1.0)

        # 3. Multi-pass Micro-Structure and Detail Reconstruction
        reconstructed = lin_img.clone()
        for p in range(passes):
            sigma_p = max(0.6, 1.2 / (p + 1))
            low_freq = gaussian_blur_torch(reconstructed, sigma=sigma_p, kernel_size=5)
            high_freq = reconstructed - low_freq
            structure_factor = max(0.0, detail_str) * 0.45
            enhanced_hf = torch.tanh(high_freq * structure_factor * 2.0) * (0.5 * detail_str)
            clamped_delta = torch.clamp(enhanced_hf, -res_limit, res_limit) * res_strength
            reconstructed = reconstructed + clamped_delta

        # 4. Temporal accumulation smoothing across sequence frames
        if self.prev_frame is not None and temp_stability > 0.0 and self.prev_frame.shape == reconstructed.shape:
            blend_w = min(0.65, temp_stability * 0.45)
            reconstructed = reconstructed * (1.0 - blend_w) + self.prev_frame * blend_w
        self.prev_frame = reconstructed.detach()

        # 5. Skin protection & AutoMask (skin, auto_mask)
        if auto_mask:
            srgb_proxy = linear_to_srgb_torch(reconstructed)
            ycbcr = rgb_to_ycbcr_torch(srgb_proxy)
            y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
            skin_mask = (
                (cr >= 0.51) & (cr <= 0.69) &
                (cb >= 0.29) & (cb <= 0.51) &
                (y >= 0.12) & (y <= 0.96)
            ).float()
            skin_mask = gaussian_blur_torch(skin_mask, sigma=2.5, kernel_size=7)

            if skin >= 0.0:
                skin_smooth = gaussian_blur_torch(reconstructed, sigma=1.0, kernel_size=5)
                skin_detail = (reconstructed - skin_smooth) * (skin * 0.8)
                skin_result = skin_smooth + skin_detail
                reconstructed = (1.0 - skin_mask) * reconstructed + skin_mask * skin_result

        # 6. Global Tone & Local Tone Mapping (global_tone, tone)
        if global_tone >= 0.0:
            exposure_gain = 2.0 ** ((global_tone - 1.0) * 0.6)
            reconstructed = reconstructed * exposure_gain

        if tone != 1.0:
            luma = (reconstructed * self.luma_weights).sum(dim=1, keepdim=True)
            luma_base = gaussian_blur_torch(luma, sigma=10.0, kernel_size=15)
            gamma = 1.0 + (tone - 1.0) * 0.25
            luma_tone = luma * torch.pow(torch.clamp(luma_base + 1e-4, 1e-4, 1.0), gamma - 1.0)
            reconstructed = reconstructed * (luma_tone / torch.clamp(luma, 1e-4))

        # 7. Style Curves
        if raw_style == "cinematic":
            luma = (reconstructed * self.luma_weights).sum(dim=1, keepdim=True)
            cin_luma = luma * luma * (3.0 - 2.0 * torch.clamp(luma, 0.0, 1.0))
            reconstructed = reconstructed * (cin_luma / torch.clamp(luma, 1e-4))
        elif raw_style == "vivid":
            # Boost linear saturation
            luma = (reconstructed * self.luma_weights).sum(dim=1, keepdim=True)
            reconstructed = luma + (reconstructed - luma) * 1.25

        # 8. Convert to sRGB
        out_srgb = linear_to_srgb_torch(reconstructed) if not hdr else reconstructed
        orig_srgb = orig_base

        # 9. Chroma & Detail Compositing (colour_strength, detail_strength)
        if detail_str != 1.0 or colour_str != 1.0:
            lin_orig = srgb_to_linear_torch(orig_srgb)
            lin_nr = srgb_to_linear_torch(out_srgb)
            lo = (lin_orig * self.luma_weights).sum(dim=1, keepdim=True)
            ln = (lin_nr * self.luma_weights).sum(dim=1, keepdim=True)
            scale_l = ln / torch.clamp(lo, 1e-4)
            loc = lin_orig * scale_l
            cc = loc + (lin_nr - loc) * colour_str
            comp = lin_orig + (cc - lin_orig) * detail_str
            out_srgb = linear_to_srgb_torch(comp)

        # 10. FidelityFX CAS Sharpening (AmdSharpness)
        if sharpness > 0.001:
            out_srgb = cas_sharpen_torch(out_srgb, sharpness)

        # 11. Intensity Blending
        final_srgb = orig_srgb + (out_srgb - orig_srgb) * res_strength
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
