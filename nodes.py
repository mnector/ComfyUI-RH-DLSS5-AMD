# SPDX-License-Identifier: MIT
# ComfyUI-RH-DLSS5: single-node DLSS 5 Neural Rendering (NGX feature 18)
# enhance / upscale for IMAGE and VIDEO, Windows (in-process bridge) and
# Linux (Wine host) backends.

from __future__ import annotations

import hashlib
import os
import platform
from typing import Optional

import numpy as np
import torch

from .dlss5nr import common
from .dlss5nr.common import DLSS5Error
from .dlss5nr.decoder import ChunkedVideoReader, build_source
from .dlss5nr.motion import TemporalGuideGenerator, sample_u8

_UPSCALE_MODES = [
    "1x (DLAA / native)",
    "1.5x (Quality)",
    "1.724x (Balanced)",
    "2x (Performance)",
    "3x (Ultra Performance)",
    "1K",
    "2K",
    "4K",
    "8K",
]

_SCALE_KEY = {"1x": 1.0, "1.5x": 1.5, "1.724x": 1.724, "2x": 2.0, "3x": 3.0}


def _load_backend(name: str):
    if name == "windows-bridge":
        from .dlss5nr import windows_backend

        if not windows_backend.supported():
            raise DLSS5Error("The windows-bridge backend requires Windows 10/11 x64.")
        return windows_backend
    from .dlss5nr import linux_backend

    if platform.system() == "Windows":
        raise DLSS5Error(
            "The linux-wine backend requires Linux; on Windows use backend=windows-bridge."
        )
    return linux_backend


def _pick_backend(backend: str) -> str:
    if backend == "auto":
        return "windows-bridge" if platform.system() == "Windows" else "linux-wine"
    return backend


def _channel_choice(frame: np.ndarray, reference: np.ndarray, order: str) -> tuple[np.ndarray, str]:
    """Detect R/B swap of the NR output (runtime builds differ). Adapted from
    the MIT-licensed ComfyUI-DLSS5-NR-Linux frontend."""
    if order == "RGBA":
        return frame, "RGBA"
    if order == "BGRA":
        return frame[..., ::-1], "BGRA"
    if reference.ndim == 3 and reference.shape[-1] > 3:
        # RGBA input: the NR output is always 3-channel, so the alpha channel
        # would fail the broadcast below (GH issue #5).
        reference = reference[..., :3]
    sample_h = min(128, frame.shape[0], reference.shape[0])
    sample_w = min(128, frame.shape[1], reference.shape[1])
    iy = np.linspace(0, frame.shape[0] - 1, sample_h, dtype=np.int64)
    ix = np.linspace(0, frame.shape[1] - 1, sample_w, dtype=np.int64)
    ry = np.linspace(0, reference.shape[0] - 1, sample_h, dtype=np.int64)
    rx = np.linspace(0, reference.shape[1] - 1, sample_w, dtype=np.int64)
    raw = frame[np.ix_(iy, ix)]
    ref = reference[np.ix_(ry, rx)]
    swapped = raw[..., ::-1]
    raw_score = float(np.mean(np.abs(raw - ref))) + float(
        np.mean(np.abs(raw.mean(axis=(0, 1)) - ref.mean(axis=(0, 1)))))
    swap_score = float(np.mean(np.abs(swapped - ref))) + float(
        np.mean(np.abs(swapped.mean(axis=(0, 1)) - ref.mean(axis=(0, 1)))))
    if raw_score <= swap_score:
        return frame, "RGBA"
    return frame[..., ::-1], "BGRA"


def _build_params(kwargs: dict) -> dict:
    mode = kwargs.get("upscaling_mode") or kwargs.get("Upscaling Mode") or _UPSCALE_MODES[0]
    backend = kwargs.get("backend") or kwargs.get("amd_backend") or "amd_nr_bridge"
    is_zluda = (backend == "zluda_tensor")

    runtime = common.resolve_runtime_dir(kwargs.get("runtime_dir", "") or "", allow_fallback=is_zluda)
    mode_name = str(mode).strip()
    auto_bucket = mode_name if mode_name in common.AUTO_BUCKETS else None
    if auto_bucket is not None:
        common.check_runtime_files(runtime, 1.0, is_zluda=is_zluda)
    else:
        mode_str = str(mode).lower()
        if "1.5" in mode_str or "quality" in mode_str:
            scale = 1.5
        elif "1.724" in mode_str or "balanced" in mode_str or "balance" in mode_str:
            scale = 1.724
        elif "3x" in mode_str or "ultra" in mode_str:
            scale = 3.0
        elif "2x" in mode_str or "performance" in mode_str:
            scale = 2.0
        else:
            scale = 1.0
        common.check_runtime_files(runtime, scale, is_zluda=is_zluda)

    nr_style = kwargs.get("nr_style") or kwargs.get("NR style") or "Default"
    presets = {
        "Cinematic": {"residual_strength": 0.85, "residual_limit": 0.20, "temporal_stability": 0.75, "sharpening": 0.00, "detail_strength": 1.00, "colour_strength": 1.00},
        "Crisp":     {"residual_strength": 1.20, "residual_limit": 0.35, "temporal_stability": 0.50, "sharpening": 0.30, "detail_strength": 1.00, "colour_strength": 1.00},
        "Natural":   {"residual_strength": 0.70, "residual_limit": 0.15, "temporal_stability": 0.70, "sharpening": 0.00, "detail_strength": 1.00, "colour_strength": 1.00},
        "Vivid":     {"residual_strength": 1.10, "residual_limit": 0.30, "temporal_stability": 0.60, "sharpening": 0.15, "detail_strength": 1.00, "colour_strength": 1.00},
        "Default":   {"residual_strength": 1.00, "residual_limit": 0.25, "temporal_stability": 0.60, "sharpening": 0.00, "detail_strength": 1.00, "colour_strength": 1.00},
    }
    preset_vals = presets.get(nr_style, {})

    enable_nr = bool(kwargs.get("enable_nr", kwargs.get("Enable Neural Rendering", True)))
    neural_runtime = str(kwargs.get("neural_runtime", kwargs.get("Neural runtime", "danielblnc 0.5.1")))
    neural_placement = str(kwargs.get("neural_placement", kwargs.get("Neural placement", "Before upscaling")))
    nr_resolution = int(kwargs.get("nr_resolution", kwargs.get("NR resolution (%)", 100)))
    neural_passes = int(kwargs.get("neural_passes", kwargs.get("Neural passes", 1)))
    full_network = bool(kwargs.get("full_network", kwargs.get("Full network (all 71 blocks)", False)))
    fast_mode = bool(kwargs.get("fast_mode", kwargs.get("Fast mode", False)))
    dynamic_nr_resolution = bool(kwargs.get("dynamic_nr_resolution", kwargs.get("Dynamic NR resolution", False)))
    model_interleave = str(kwargs.get("model_interleave", kwargs.get("Model interleave", "Off")))

    residual_strength = float(kwargs.get("residual_strength", preset_vals.get("residual_strength", 1.00)))
    residual_limit = float(kwargs.get("residual_limit", preset_vals.get("residual_limit", 0.25)))
    temporal_stability = float(kwargs.get("temporal_stability", preset_vals.get("temporal_stability", 0.60)))
    sharpening = float(kwargs.get("sharpening", preset_vals.get("sharpening", 0.00)))
    detail_strength = float(kwargs.get("detail_strength", preset_vals.get("detail_strength", 1.00)))
    colour_strength = float(kwargs.get("colour_strength", preset_vals.get("colour_strength", 1.00)))
    colour_composition = str(kwargs.get("colour_composition", kwargs.get("Colour composition", "Classic")))

    try:
        gpu_index = int(os.environ.get("DLSS5_GPU_INDEX", "0") or 0)
    except ValueError:
        gpu_index = 0

    return {
        "runtime": runtime,
        "is_zluda": is_zluda,
        "backend": backend,
        "amd_backend": backend,
        "gpu_index": gpu_index,
        "scale": None if auto_bucket is not None else scale,
        "auto_bucket": auto_bucket,
        "enable_nr": enable_nr,
        "neural_runtime": neural_runtime,
        "nr_style": nr_style,
        "neural_placement": neural_placement,
        "nr_resolution": nr_resolution,
        "neural_passes": neural_passes,
        "full_network": full_network,
        "fast_mode": fast_mode,
        "dynamic_nr_resolution": dynamic_nr_resolution,
        "model_interleave": model_interleave,
        "residual_strength": residual_strength,
        "residual_limit": residual_limit,
        "temporal_stability": temporal_stability,
        "sharpening": sharpening,
        "colour_composition": colour_composition,
        "detail_strength": detail_strength,
        "colour_strength": colour_strength,
        "warmup_frames": 1,
        "keep_audio": kwargs.get("keep_audio", "on") == "on",
        "motion_mode": kwargs.get("motion", "auto"),
        "scene_change_threshold": float(kwargs.get("scene_change_threshold", 0.24)),
        "batch_temporal": True,
        "channel_order": "auto",
        # Internal fallbacks for compatibility
        "style": 0,
        "intensity": residual_strength,
        "structure": detail_strength,
        "tone": 1.0,
        "skin": -1.0,
        "auto_mask": True,
        "hdr": False,
        "global_tone": -1.0,
        "detail": detail_strength,
        "color": colour_strength,
    }


def _resolve_auto_scale(params: dict, width: int, height: int) -> tuple[dict, str]:
    """Resolve an auto-bucket upscaling mode against concrete input dimensions.

    Returns (possibly-copied params, human-readable note). Fixed-factor modes
    pass through unchanged with an empty note.
    """
    bucket = params.get("auto_bucket")
    if not bucket:
        return params, ""
    params = dict(params)
    params["scale"] = common.auto_scale_factor(width, height, bucket)
    if params["scale"] > 1.0 and not params.get("is_zluda", False):
        common.check_runtime_files(params["runtime"], params["scale"])
    return params, (f"auto {bucket}: short edge {min(int(width), int(height))}->"
                    f"{common.AUTO_BUCKETS[bucket]} via {params['scale']:g}x")


def _make_nvof_generator(width: int, height: int, threshold: float):
    try:
        from .dlss5nr.nvof import NvofGuideGenerator
    except Exception:
        return None
    return NvofGuideGenerator(width, height, threshold)


def _select_motion_engine(pixels_u8: np.ndarray, params: dict):
    """Pick the guide generator for this run.

    Returns (generator, engine_note). The generator is None when only zero
    motion is possible; the note names the engine actually used, including
    any fallback that happened, and goes into the node's status output.
    """
    mode = params["motion_mode"]
    if TemporalGuideGenerator is None:
        return None, "none"
    height, width = pixels_u8.shape[:2]
    threshold = params["scene_change_threshold"]
    from .dlss5nr.amd_engine import is_amd_environment
    if is_amd_environment() and mode == "auto":
        # On AMD GPUs, automatically select DIS without checking or warning about NVOFA
        generator = TemporalGuideGenerator(width, height, threshold)
        if generator.available:
            return generator, "dis"
        return None, "none"

    nvof_note = ""
    if mode in ("auto", "nvof"):
        generator = _make_nvof_generator(width, height, threshold)
        if generator is not None and generator.available:
            return generator, "nvof"
        if not is_amd_environment():
            nvof_note = (
                f"nvof unavailable ({generator.init_error})"
                if generator is not None
                else "nvof unavailable (module import failed)"
            )
            print(f"[RH-DLSS5] {nvof_note}; falling back to DIS optical flow", flush=True)
    generator = TemporalGuideGenerator(width, height, threshold)
    if generator.available:
        return generator, "nvof->optical_flow" if nvof_note else "optical_flow"
    if nvof_note:
        print("[RH-DLSS5] DIS optical flow unavailable too; zero motion", flush=True)
    return None, "none"


def _frame_source_iter(frame_iter, params: dict, holder=None):
    """Yield (rgb_u8, motion_fp16, reset) over an iterator of uint8 frames.

    Shared by the tensor path (frames sampled from a float batch) and the
    chunked-decode path (frames streamed from an ffmpeg pipe). `holder` lets
    the streaming Enhance loop access the frame currently being processed for
    channel-order detection without buffering the clip.
    """
    temporal = params["batch_temporal"]
    flow_requested = params["motion_mode"] in ("auto", "nvof", "optical_flow")
    guides = None
    zero_motion: Optional[np.ndarray] = None
    seen_first = False
    for pixels_u8 in frame_iter:
        if holder is not None:
            holder.current = pixels_u8
        if flow_requested and guides is None:
            guides, engine_note = _select_motion_engine(pixels_u8, params)
            params["motion_engine"] = engine_note
        if guides is not None and guides.available:
            motion, guide_reset = guides.process(pixels_u8)
            reset = guide_reset or not temporal
        else:
            if zero_motion is None:
                zero_motion = np.zeros((pixels_u8.shape[0], pixels_u8.shape[1], 2), dtype=np.float16)
            motion = zero_motion
            reset = (not temporal) or (not seen_first)
            seen_first = True
        yield pixels_u8, motion, reset


def _frame_source(images: torch.Tensor, params: dict):
    """Yield (rgb_u8, motion_fp16, reset) for every input frame."""
    return _frame_source_iter(
        (sample_u8(images[index].detach().cpu().numpy()) for index in range(images.shape[0])),
        params)


def _make_progress_bar(total: int = 100):
    try:
        from comfy.utils import ProgressBar
        return ProgressBar(total)
    except Exception:
        class _DummyBar:
            def update_absolute(self, val): pass
        return _DummyBar()


def _progress_callback():
    try:
        bar = _make_progress_bar(100)

        def callback(current: int, frames_total: int) -> None:
            bar.update_absolute(int(100 * current / max(1, frames_total)))

        return callback
    except Exception:
        return None


class RH_DLSS5Enhance_AMD:
    """DLSS 5 Neural Rendering enhance/upscale node adapted for AMD GPUs.

    IMAGE in -> IMAGE out, VIDEO in -> VIDEO out. Connect the outputs to the
    standard SaveImage / SaveVideo nodes. Only one input needs to be connected;
    if both are connected both are processed.
    """

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 (ComfyUI convention)
        return {
            "required": {},
            "optional": {
                # Media inputs: only image or video (NO depth, normal, or albedo inputs required)
                "image": ("IMAGE", {"tooltip": "Optional image batch (e.g. LoadImage). Each batch entry is one frame in temporal order."}),
                "video": ("VIDEO", {"tooltip": "Optional video (e.g. LoadVideo). Frames are processed in temporal order with chunked streaming."}),

                # AMD-NR Core
                "enable_nr": ("BOOLEAN", {"default": True, "tooltip": "Enable DLSS 5 Neural Rendering (DLSS-NR on AMD)."}),
                "neural_runtime": (["danielblnc 0.5.1", "lmxxf"], {"default": "danielblnc 0.5.1", "tooltip": "Neural runtime engine for AMD (danielblnc native HIP kernels or lmxxf runtime)."}),
                "upscaling_mode": (_UPSCALE_MODES, {"default": _UPSCALE_MODES[0], "tooltip": "Upscaling carrier resolution (1x enhances at native size; fixed factors upscale with FSR/DLSS before/after neural pass; 1K/2K/4K/8K auto-fit short edge)."}),
                "nr_style": (["Default", "Cinematic", "Crisp", "Natural", "Vivid", "Custom"], {"default": "Default", "tooltip": "AMD-NR style tuning presets: Cinematic (deep contrast/shadows), Crisp (maximum edge definition & CAS clarity), Natural (faithful micro-textures), Vivid (punchy chroma/highlights), Default, or Custom (uses slider values below)."}),

                # Performance
                "neural_placement": (["Before upscaling", "After upscaling"], {"default": "Before upscaling", "tooltip": "Run neural rendering before or after the upscaling carrier pass."}),
                "nr_resolution": ("INT", {"default": 100, "min": 10, "max": 200, "step": 5, "tooltip": "NR working resolution percentage (100% = full render size)."}),
                "neural_passes": ("INT", {"default": 1, "min": 1, "max": 3, "step": 1, "tooltip": "Number of neural refinement passes (1 = default fast pass, 2-3 = deeper reconstruction)."}),
                "full_network": ("BOOLEAN", {"default": False, "tooltip": "Run full 71-block neural network (lmxxf runtime only)."}),
                "fast_mode": ("BOOLEAN", {"default": False, "tooltip": "Enable fast execution mode for higher throughput."}),
                "dynamic_nr_resolution": ("BOOLEAN", {"default": False, "tooltip": "Dynamic neural rendering resolution scaling."}),
                "model_interleave": (["Off", "On"], {"default": "Off", "tooltip": "Interleave neural model evaluations across frames."}),

                # Quality
                "residual_strength": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Residual transfer strength: Controls how much high-frequency reconstructed detail is transferred to the image."}),
                "residual_limit": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Residual clamp ratio: Limits residual amplification to prevent haloing/clipping."}),
                "temporal_stability": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Temporal stability across sequential frames to suppress flicker."}),
                "sharpening": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "FidelityFX CAS (Contrast Adaptive Sharpening) strength."}),

                # Image look
                "colour_composition": (["Classic", "RenoDX (experimental)"], {"default": "Classic", "tooltip": "Colour composition mode."}),
                "detail_strength": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Local detail and micro-structure strength."}),
                "colour_strength": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Chroma strength and color fidelity."}),

                # Motion, Audio & Backend
                "motion": (["auto", "optical_flow", "none"], {"default": "auto", "tooltip": "Motion vector generation: auto and optical_flow compute dense motion via OpenCV DIS internally without user input."}),
                "keep_audio": (["on", "off"], {"default": "on", "tooltip": "Preserve original audio in VIDEO output."}),
                "backend": (["amd_nr_bridge", "zluda_tensor"], {"default": "amd_nr_bridge", "tooltip": "Execution backend: amd_nr_bridge runs the real DLSS-NR neural network via D3D12/OptiScaler (danielblnc HIP kernels / lmxxf runtime); zluda_tensor is a pure PyTorch GPU heuristic fallback."}),
                "runtime_dir": ("STRING", {"default": "", "tooltip": "Optional custom path to DLSS5/AMD-NR runtime folder (empty = ComfyUI/models/dlss5/)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO", "STRING")
    RETURN_NAMES = ("images", "video", "status")
    FUNCTION = "enhance"
    CATEGORY = "image/upscaling [AMD]"
    DESCRIPTION = (
        "Runs DLSS 5 Neural Rendering (NGX feature 18) and neural upscaling on AMD GPUs "
        "(accelerated via ZLUDA/ROCm or D3D12/HIP bridge with OptiScaler / danielblnc DLSS-NR-on-AMD). "
        "Supports IMAGE and VIDEO with chunked streaming and audio preservation."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        digest = hashlib.sha256()
        for key in sorted(kwargs):
            if key in ("image", "video", "Image"):
                continue
            digest.update(f"{key}={kwargs.get(key)!r};".encode())
        image = kwargs.get("image") if kwargs.get("image") is not None else kwargs.get("Image")
        if image is not None:
            tensor = image if isinstance(image, torch.Tensor) else image[0]
            digest.update(str(tuple(tensor.shape)).encode())
            flat = tensor.detach().reshape(-1)[:: max(1, tensor.numel() // 4096)]
            digest.update(flat.to(torch.float16).cpu().numpy().tobytes())
        video = kwargs.get("video")
        if video is not None:
            source = _audio_source_path_of(video)
            if source:
                try:
                    stat = os.stat(source)
                    digest.update(f"{source}:{stat.st_size}:{stat.st_mtime}".encode())
                except OSError:
                    digest.update(source.encode())
            else:
                digest.update(repr(video).encode())
        return digest.hexdigest()

    def _bypass_passthrough(self, image, video, kwargs) -> tuple[Optional[torch.Tensor], object, str]:
        """style=off: neural rendering disabled, frames pass through untouched."""
        from .dlss5nr.videoout import build_video

        note = "NR bypassed (style=off): frames passed through unprocessed"
        out_image: Optional[torch.Tensor] = None
        out_video = None
        if image is not None:
            out_image = image
            video_obj, _ = build_video(image, 24.0, None, None)
            if video_obj is not None:
                out_video = video_obj
            note = f"IMAGE {int(image.shape[0])} frames; {note}"
        if video is not None:
            src_path = _audio_source_path_of(video)
            if src_path:
                # File-backed: pass the VIDEO object through untouched. Full
                # decoding just to re-encode identical frames would OOM on
                # long/4K clips (the 20260914 incident pattern).
                out_video = video
                note = f"VIDEO bypassed (style=off): file passed through unprocessed ({src_path})"
            else:
                components = video.get_components()
                frames = components.images
                config = getattr(components, "config", None)
                fps = float(getattr(config, "fps", 24.0) or 24.0) if config is not None else 24.0
                video_obj, _ = build_video(
                    frames, fps, components.audio if str(kwargs.get("keep_audio", "on")) == "on" else None,
                    video,
                )
                out_video = video_obj
                note = f"VIDEO {int(frames.shape[0])} frames @{fps:g}fps; {note}"
        return out_image, out_video, note

    def enhance(self, **kwargs) -> tuple[Optional[torch.Tensor], object, str]:
        image = kwargs.get("image") if kwargs.get("image") is not None else kwargs.get("Image")
        video = kwargs.get("video")
        if image is None and video is None:
            raise DLSS5Error("The enhance node needs at least one input: connect `image` or `video`.")
        enable_nr = kwargs.get("enable_nr", kwargs.get("Enable Neural Rendering", True))
        if not enable_nr or str(kwargs.get("nr_style", "")).startswith("off") or str(kwargs.get("style", "")).startswith("off"):
            return self._bypass_passthrough(image, video, kwargs)
        params = _build_params(kwargs)
        is_zluda = params.get("is_zluda", False)
        backend_name = "zluda-tensor" if is_zluda else _pick_backend(kwargs.get("backend", "auto"))
        gt_note = f", gt={params['global_tone']:g}" if params["global_tone"] >= 0 else ""

        out_image: Optional[torch.Tensor] = None
        out_video = None
        notes: list[str] = []

        if image is not None:
            img_params, auto_note = _resolve_auto_scale(params, int(image.shape[2]), int(image.shape[1]))
            if is_zluda:
                result, iw, ih, ow, oh, order, engine = _run_frames_zluda(image, img_params)
            else:
                result, iw, ih, ow, oh, order, engine = _run_frames(image, img_params, backend_name)
            out_image = result
            notes.append(
                f"IMAGE {int(image.shape[0])} frames {iw}x{ih}->{ow}x{oh} via {backend_name} "
                f"(channel {order}, motion {engine}{', hdr' if params['hdr'] else ''}{gt_note})"
                + (f"; {auto_note}" if auto_note else "")
            )

        if video is not None:
            src_path = _audio_source_path_of(video)
            reader = None
            if src_path:
                try:
                    reader = ChunkedVideoReader(src_path)
                except Exception as exc:
                    print(f"[RH-DLSS5-AMD] chunked decode unavailable ({exc}); full decode", flush=True)
            if reader is not None:
                # ---- 文件直传: 分块解码 + 逐帧流式编码, 内存峰值 O(块) ----
                from .dlss5nr.fg import as_fps
                fps = as_fps(reader.fps if reader.fps > 0 else 24.0)
                print(f"[RH-DLSS5-AMD] Enhance chunked decode: {reader.count} frames "
                      f"{reader.width}x{reader.height} @{reader.fps:g}fps from {src_path}", flush=True)
                vid_params, auto_note = _resolve_auto_scale(params, int(reader.width), int(reader.height))
                keep_audio = params["keep_audio"]
                if is_zluda:
                    video_obj, iw, ih, ow, oh, order, engine = _run_frames_stream_zluda(
                        reader, fps, vid_params, source_video=video if keep_audio else None
                    )
                else:
                    video_obj, iw, ih, ow, oh, order, engine = _run_frames_stream(
                        reader, fps, vid_params, backend_name,
                        source_video=video if keep_audio else None)
                out_video = video_obj
                notes.append(
                    f"VIDEO {reader.count} frames @{float(fps):g}fps {iw}x{ih}->{ow}x{oh} via {backend_name} "
                    f"(chunked decode, streaming encode, channel {order}, motion {engine}"
                    f"{', hdr' if params['hdr'] else ''}{gt_note})"
                    + (f"; {auto_note}" if auto_note else "")
                )
            else:
                components = video.get_components()
                frames = components.images
                config = getattr(components, "config", None)
                fps = float(getattr(config, "fps", 24.0) or 24.0) if config is not None else 24.0
                keep_audio = params["keep_audio"]
                vid_params, auto_note = _resolve_auto_scale(params, int(frames.shape[2]), int(frames.shape[1]))
                if is_zluda:
                    result, iw, ih, ow, oh, order, engine = _run_frames_zluda(frames, vid_params)
                else:
                    result, iw, ih, ow, oh, order, engine = _run_frames(frames, vid_params, backend_name)
                from .dlss5nr.videoout import build_video

                video_obj, note = build_video(
                    result, fps, components.audio if keep_audio else None, video
                )
                out_video = video_obj
                notes.append(
                    f"VIDEO {int(frames.shape[0])} frames @{fps:g}fps {iw}x{ih}->{ow}x{oh} via {backend_name} "
                    f"(channel {order}, motion {engine}{', hdr' if params['hdr'] else ''}{gt_note}); {note}"
                    + (f"; {auto_note}" if auto_note else "")
                )

        return out_image, out_video, " | ".join(notes)


class RH_DLSS5FrameInterpolation_AMD:
    """DLSS Frame Generation and OpenCV DIS temporal interpolation for AMD GPUs.

    Streams frames to the dlssg-worker.exe NGX host or falls back to GPU-accelerated
    bidirectional OpenCV DIS optical flow with pyramidal warping in PyTorch/ZLUDA.
    """

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 (ComfyUI convention)
        return {
            "required": {},
            "optional": {
                "output_fps": (["1x", "2x", "3x", "4x", "23.976", "24", "25", "29.97", "30", "48", "50", "59.94", "60", "72", "90", "96", "120", "144"], {"default": "2x", "tooltip": "1x = passthrough: the source is returned untouched (no interpolation, no re-encode; the VIDEO object is passed through as-is, motion/keep_audio options are ignored). Either a frame-rate multiplier (2x/3x/4x: DLSS generates one frame per interval, 3x/4x cascade 2x passes) or an exact target output frame rate. For a target rate the node cascades DLSS 2x passes into a dense CFR grid (2/4/8x source) and nearest-picks the target timeline, so e.g. a 24fps source at 60fps interpolates a 96fps grid and picks every ~1.6th frame (1-2-1-2 cadence, no duplicated frames). Target rates must exceed source fps and be at most 6x source; intermediate grid is at most 8x. No tail extension."}),
                "video": ("VIDEO", {"tooltip": "Optional video (e.g. LoadVideo). Source frame rate and audio come from this input."}),
                "image": ("IMAGE", {"tooltip": "Optional image batch (e.g. LoadImage). Each batch entry is one frame in temporal order; batches are treated as a 24fps source."}),
                "motion": (["auto", "nvof", "dis"], {"default": "auto", "tooltip": "Motion vectors guiding interpolation. auto = hardware NVOFA optical flow when available, else OpenCV DIS (auto-selected on AMD without errors); nvof = hardware NVOFA only; dis = OpenCV DIS. Scene cuts hold the previous frame in each interpolation slot."}),
                "scene_change_threshold": ("FLOAT", {"default": 0.24, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Mean luminance change above which temporal history resets (scene cut detection). Higher = fewer resets."}),
                "keep_audio": (["on", "off"], {"default": "on", "tooltip": "Keep the original audio in the VIDEO output. Audio is stream-copied from the source file; when no source file is available it is re-encoded from the AUDIO input instead."}),
                "audio": ("AUDIO", {"tooltip": "Optional audio for the VIDEO output. With a VIDEO input it replaces the source audio while keep_audio is on; with an IMAGE input it is the only way to attach sound. Re-encoded to AAC when it cannot be stream-copied."}),
                "runtime_dir": ("STRING", {"default": "", "tooltip": "Optional override of the folder holding dlssg-worker.exe + nvngx.dll + _nvngx.dll + nvngx_dlssg.dll. Empty uses DLSS5_FG_RUNTIME_DIR, then <DLSS5 runtime>/dlssg, then the plugin's runtime/dlssg folder."}),
                "wine_prefix": ("STRING", {"default": "", "tooltip": "Linux only: WINEPREFIX for the worker (ignored on Windows, where the worker runs natively). Empty uses DLSS5_WINEPREFIX, then ~/.wine; validated at run time (needs DXVK + DXVK-NVAPI installed in the prefix)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO", "STRING")
    RETURN_NAMES = ("images", "video", "status")
    FUNCTION = "interpolate"
    CATEGORY = "video [AMD]"
    DESCRIPTION = (
        "Raises a video's frame rate on AMD GPUs using DLSS Frame Generation or high-speed "
        "OpenCV DIS bidirectional optical flow with GPU pyramidal warping in PyTorch/ZLUDA. "
        "Supports 2x/3x/4x multipliers or exact target frame rates with audio preservation."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        digest = hashlib.sha256()
        for key in sorted(kwargs):
            if key in ("image", "video", "audio"):
                continue
            digest.update(f"{key}={kwargs.get(key)!r};".encode())
        image = kwargs.get("image")
        if image is not None:
            tensor = image if isinstance(image, torch.Tensor) else image[0]
            digest.update(str(tuple(tensor.shape)).encode())
            flat = tensor.detach().reshape(-1)[:: max(1, tensor.numel() // 4096)]
            digest.update(flat.to(torch.float16).cpu().numpy().tobytes())
        audio = kwargs.get("audio")
        if audio is not None:
            wave = audio.get("waveform") if isinstance(audio, dict) else getattr(audio, "waveform", None)
            rate = audio.get("sample_rate") if isinstance(audio, dict) else getattr(audio, "sample_rate", None)
            if wave is not None:
                tensor = torch.as_tensor(wave)
                digest.update(f"audio:{rate}:{tuple(tensor.shape)}".encode())
                flat = tensor.detach().reshape(-1)[:: max(1, tensor.numel() // 4096)]
                digest.update(flat.to(torch.float16).cpu().numpy().tobytes())
        video = kwargs.get("video")
        if video is not None:
            source = _audio_source_path_of(video)
            if source:
                try:
                    stat = os.stat(source)
                    digest.update(f"{source}:{stat.st_size}:{stat.st_mtime}".encode())
                except OSError:
                    digest.update(source.encode())
            else:
                digest.update(repr(video).encode())
        return digest.hexdigest()

    def interpolate(self, **kwargs) -> tuple[Optional[torch.Tensor], object, str]:
        import gc

        from .dlss5nr.fg import (MULTIPLIERS, as_fps, interpolate_frames, interpolate_stream,
                         interpolate_stream_from_source)
        from .dlss5nr.videoout import StreamEncoder, build_video, video_from_file

        image = kwargs.get("image")
        video = kwargs.get("video")
        if image is None and video is None:
            raise DLSS5Error("The frame interpolation node needs at least one input: connect `image` or `video`.")
        choice = str(kwargs.get("output_fps", "2x") or "2x")
        if choice == "1x":
            # 1x passthrough; the real handling is the early-return branch below
            # (after out_image/out_video/notes are initialized).
            multiplier, target_fps = 1, None
        elif choice in MULTIPLIERS:
            multiplier, target_fps = MULTIPLIERS[choice], None
        else:
            multiplier, target_fps = 2, as_fps(choice)
        motion_mode = str(kwargs.get("motion", "auto"))
        threshold = float(kwargs.get("scene_change_threshold", 0.24) or 0.24)
        keep_audio = str(kwargs.get("keep_audio", "on")) == "on"
        audio_input = kwargs.get("audio")
        runtime_dir = str(kwargs.get("runtime_dir", "") or "")
        wine_prefix = str(kwargs.get("wine_prefix", "") or "")

        out_image: Optional[torch.Tensor] = None
        out_video = None
        notes: list[str] = []

        if choice == "1x":
            # 1x passthrough: no interpolation, no re-encode. The VIDEO object
            # is returned as-is; the image batch is rebuilt into a video only
            # for the video output slot, at the source frame rate.
            if image is not None:
                src_fps = as_fps(kwargs.get("images_fps", 24.0))
                out_image = image
                notes.append(f"IMAGE {int(image.shape[0])} frames @{float(src_fps):g}fps, "
                             f"1x passthrough (no interpolation)")
                video_obj, vnote = build_video(image, src_fps, kwargs.get("audio"), None)
                if video_obj is not None:
                    out_video = video_obj
                if vnote:
                    notes.append(vnote)
            if video is not None:
                out_video = video
                notes.append("VIDEO 1x passthrough: source frames and audio unchanged "
                             "(no interpolation)")
            return out_image, out_video, " | ".join(notes)

        if image is not None:
            src_fps = as_fps(kwargs.get("images_fps", 24.0))
            bar = _make_progress_bar(100)
            result, out_fps, note = interpolate_frames(
                image, src_fps, multiplier, motion_mode, threshold, runtime_dir, wine_prefix,
                progress_callback=lambda done, total: bar.update_absolute(100 * done // max(1, total)),
                target_fps=target_fps,
            )
            out_image = result
            notes.append(f"IMAGE {int(image.shape[0])}->{int(result.shape[0])} frames "
                         f"@{float(src_fps):g}->{float(out_fps):g}fps; {note}")
            video_obj, vnote = build_video(result, out_fps, audio_input, None)
            if video_obj is not None:
                out_video = video_obj
            if vnote:
                notes.append(vnote)

        if video is not None:
            chunked = None
            src_path = _audio_source_path_of(video)
            if src_path:
                chunked = build_source(src_path)   # (reader, SequentialSource) 或 None
            if chunked is not None:
                # ---- 文件直传: 分块解码输入, 内存峰值 O(块) ----
                reader, source = chunked
                src_fps = as_fps(reader.fps if reader.fps > 0 else 24)
                print(f"[RH-DLSS5-AMD] FG chunked decode: {reader.count} frames "
                      f"{reader.width}x{reader.height} @{reader.fps:g}fps from {src_path}", flush=True)
                if target_fps is not None and target_fps <= src_fps:
                    raise DLSS5Error(
                        f"target output fps ({float(target_fps):g}) must exceed the source "
                        f"rate ({float(src_fps):g}); interpolation only increases frame rate.")
                total_in, height, width = int(reader.count), int(reader.height), int(reader.width)
                audio = audio_input if keep_audio else None
                out_fps = target_fps if target_fps is not None else src_fps * multiplier
                # audio=None + source_video=video: StreamEncoder 从源文件流拷贝音轨
                encoder = StreamEncoder(width, height, out_fps, audio,
                                        video if (audio_input is None and keep_audio) else None)
                bar = _make_progress_bar(100)
                try:
                    out_fps, note = interpolate_stream_from_source(
                        source, total_in, width, height, src_fps, multiplier,
                        motion_mode, threshold, runtime_dir, wine_prefix,
                        progress_callback=lambda done, total: bar.update_absolute(100 * done // max(1, total)),
                        frame_sink=encoder.write, target_fps=target_fps,
                    )
                    encoder.close()
                except BaseException:
                    encoder.abort()
                    raise
                out_video = video_from_file(encoder.path)
                if encoder.source:
                    audio_note = "audio stream-copied from source"
                elif audio_input is not None:
                    audio_note = "audio re-encoded from audio input"
                else:
                    audio_note = "no audio stream"
                notes.append(f"VIDEO {total_in} source frames "
                             f"@{float(src_fps):g}->{float(out_fps):g}fps, chunked decode + "
                             f"streaming encode, {audio_note}; {note}")
            else:
                components = video.get_components()
                frames = components.images
                config = getattr(components, "config", None)
                if config is not None and getattr(config, "fps", None):
                    src_fps = as_fps(config.fps)
                else:
                    frame_rate = getattr(components, "frame_rate", None)
                    src_fps = as_fps(frame_rate if frame_rate is not None else 24)
                if target_fps is not None and target_fps <= src_fps:
                    # Reject before decoding the whole clip: interpolate_stream
                    # would only validate after the uint8 decode and encoder spawn.
                    raise DLSS5Error(
                        f"target output fps ({float(target_fps):g}) must exceed the source "
                        f"rate ({float(src_fps):g}); interpolation only increases frame rate.")
                total_in = int(frames.shape[0])
                height, width = int(frames.shape[1]), int(frames.shape[2])
                if not keep_audio:
                    audio = None
                elif audio_input is not None:
                    audio = audio_input  # explicit AUDIO input replaces the source track
                else:
                    audio = components.audio

                # Convert the decoded float batch to uint8 once, in bounded chunks,
                # then release local float references: RGB uint8 is one quarter
                # of float32 RGB. Upstream owners and cascade buffers may remain.
                rgb = np.empty((total_in, height, width, 3), dtype=np.uint8)
                chunk = max(1, (1 << 26) // max(1, height * width))
                for start in range(0, total_in, chunk):
                    stop = min(total_in, start + chunk)
                    block = frames[start:stop].detach().cpu().mul(255.0).round_().clamp_(0.0, 255.0)
                    rgb[start:stop] = block.to(torch.uint8).numpy()
                frames = None
                try:
                    components.images = None
                except Exception:
                    pass
                gc.collect()

                # Stream the interpolation straight into the encoder: no full
                # float32 output tensor is ever built for the video path.
                out_fps = target_fps if target_fps is not None else src_fps * multiplier
                # With an explicit AUDIO input the source file must not win the
                # stream-copy race in _audio_source_path; drop it so the user
                # waveform is what gets encoded. keep_audio=off must also drop
                # the source file, or its audio track leaks into the output.
                encoder = StreamEncoder(width, height, out_fps, audio,
                                        video if (audio_input is None and keep_audio) else None)
                bar = _make_progress_bar(100)
                try:
                    out_fps, note = interpolate_stream(
                        rgb, src_fps, multiplier, motion_mode, threshold, runtime_dir, wine_prefix,
                        progress_callback=lambda done, total: bar.update_absolute(100 * done // max(1, total)),
                        frame_sink=encoder.write, target_fps=target_fps,
                    )
                    encoder.close()
                except BaseException:
                    encoder.abort()
                    raise
                out_video = video_from_file(encoder.path)
                if encoder.source:
                    audio_note = "audio stream-copied from source"
                elif audio_input is not None:
                    audio_note = "audio re-encoded from audio input"
                else:
                    audio_note = "no audio stream"
                notes.append(f"VIDEO {total_in} source frames "
                             f"@{float(src_fps):g}->{float(out_fps):g}fps, streaming encode, "
                             f"{audio_note}; {note}")

        return out_image, out_video, " | ".join(notes)


def _run_frames(images: torch.Tensor, params: dict, backend_name: str):
    input_h, input_w = int(images.shape[1]), int(images.shape[2])
    output_w, output_h = common.target_size(input_w, input_h, params["scale"])
    params = dict(params)
    frame_count = int(images.shape[0])
    params["frame_count"] = frame_count
    params["perf_quality"] = common.SCALE_TO_PERF_QUALITY[
        min(common.SCALE_TO_PERF_QUALITY, key=lambda c: abs(c - params["scale"]))
    ]
    backend = _load_backend(backend_name)
    frames = _frame_source(images, params)
    # Preallocate once and fill per frame: accumulating a Python list and then
    # np.stack + .to() would hold several full-size copies of long videos.
    result = torch.empty((frame_count, output_h, output_w, 3), dtype=images.dtype, device=images.device)
    selected_order = None
    progress = _progress_callback()
    for index, output in enumerate(
        backend.process_frames(frames, input_w, input_h, output_w, output_h, params, progress)
    ):
        reference = images[min(index, frame_count - 1)].detach().cpu().numpy()
        corrected, order = _channel_choice(
            output, reference, params["channel_order"] if selected_order is None else selected_order
        )
        if selected_order is None:
            selected_order = order
        corrected = np.clip(corrected, 0.0, 1.0)
        result[index].copy_(torch.from_numpy(corrected))
    return result, input_w, input_h, output_w, output_h, selected_order, params.get("motion_engine", "none")


def _run_frames_stream(reader, fps, params: dict, backend_name: str, source_video=None,
                       audio_input=None):
    """Chunked-decode VIDEO path: stream every frame through the backend and the
    encoder. Peak RAM is one chunk of input frames plus one output frame; the
    full float32 result tensor (181 GB for 1824 4K frames - the 20260914 OOM)
    is never built.
    """
    from .dlss5nr.videoout import StreamEncoder, video_from_file

    input_w, input_h = int(reader.width), int(reader.height)
    output_w, output_h = common.target_size(input_w, input_h, params["scale"])
    params = dict(params)
    frame_count = int(reader.count)
    params["frame_count"] = frame_count
    params["perf_quality"] = common.SCALE_TO_PERF_QUALITY[
        min(common.SCALE_TO_PERF_QUALITY, key=lambda c: abs(c - params["scale"]))
    ]
    backend = _load_backend(backend_name)

    encoder = StreamEncoder(output_w, output_h, fps, audio_input, source_video)
    holder = _FrameHolder()
    progress = _progress_callback()
    selected_order = None
    try:
        for index, output in enumerate(backend.process_frames(
            _frame_source_iter((f for _i, f in reader.iter_frames()), params, holder),
            input_w, input_h, output_w, output_h, params, progress
        )):
            need_ref = selected_order is None and params["channel_order"] == "auto"
            reference = (holder.current.astype(np.float32) / 255.0) if need_ref else output
            corrected, order = _channel_choice(
                output, reference,
                params["channel_order"] if selected_order is None else selected_order
            )
            selected_order = order
            corrected = np.clip(corrected, 0.0, 1.0)
            encoder.write(np.ascontiguousarray(corrected * 255.0).round().astype(np.uint8))
        encoder.close()
    except BaseException:
        encoder.abort()
        raise
    return (video_from_file(encoder.path), input_w, input_h, output_w, output_h,
            selected_order, params.get("motion_engine", "none"))


def _run_frames_zluda(images: torch.Tensor, params: dict):
    """ZLUDA / ROCm pure PyTorch GPU-accelerated execution for image batches."""
    from .dlss5nr.amd_engine import ZludaTensorEnhancer

    input_h, input_w = int(images.shape[1]), int(images.shape[2])
    output_w, output_h = common.target_size(input_w, input_h, params["scale"])
    frame_count = int(images.shape[0])
    device = images.device if images.is_cuda else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    enhancer = ZludaTensorEnhancer(device)
    result = torch.empty((frame_count, output_h, output_w, 3), dtype=images.dtype, device=images.device)
    progress = _progress_callback()
    for index in range(frame_count):
        frame = images[index].permute(2, 0, 1)
        out_frame = enhancer.process_frame(frame, output_w, output_h, params)
        result[index].copy_(out_frame.permute(1, 2, 0).to(images.device, dtype=images.dtype))
        if progress is not None:
            progress(index + 1, frame_count)
    return result, input_w, input_h, output_w, output_h, "RGB", "zluda_tensor"


def _run_frames_stream_zluda(reader, fps, params: dict, source_video=None, audio_input=None):
    """ZLUDA / ROCm chunked-decode + streaming encode for long videos with audio preservation."""
    from .dlss5nr.amd_engine import ZludaTensorEnhancer
    from .dlss5nr.videoout import StreamEncoder, video_from_file

    input_w, input_h = int(reader.width), int(reader.height)
    output_w, output_h = common.target_size(input_w, input_h, params["scale"])
    frame_count = int(reader.count)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    enhancer = ZludaTensorEnhancer(device)
    encoder = StreamEncoder(output_w, output_h, fps, audio_input, source_video)
    progress = _progress_callback()
    try:
        for index, frame_u8 in enumerate(reader.iter_frames()):
            frame_float = torch.from_numpy(frame_u8.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
            out_frame = enhancer.process_frame(frame_float, output_w, output_h, params)
            out_u8 = np.clip(np.rint(out_frame.permute(1, 2, 0).detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
            encoder.write(np.ascontiguousarray(out_u8))
            if progress is not None:
                progress(index + 1, frame_count)
        encoder.close()
    except BaseException:
        encoder.abort()
        raise
    return (video_from_file(encoder.path), input_w, input_h, output_w, output_h, "RGB", "zluda_tensor")


class _FrameHolder:
    """Keeps the frame currently flowing through the motion generator so the
    channel-order probe can compare it against the backend output."""

    def __init__(self):
        self.current = None


def _audio_source_path_of(video) -> Optional[str]:
    getter = getattr(video, "get_stream_source", None)
    if not callable(getter):
        return None
    try:
        info = getter()
    except Exception:
        return None
    if isinstance(info, (tuple, list)) and info:
        path = str(info[0])
        return path if os.path.exists(path) else None
    if isinstance(info, str) and os.path.exists(info):
        return info
    return None


NODE_CLASS_MAPPINGS = {
    "RH_DLSS5Enhance_AMD": RH_DLSS5Enhance_AMD,
    "RH_DLSS5FrameInterpolation_AMD": RH_DLSS5FrameInterpolation_AMD,
    # Backward compatibility aliases for previous Dlss5-AMD workflows
    "Dlss5AmdFsrSimple": RH_DLSS5Enhance_AMD,
    "Dlss5AmdNvidiaSimple": RH_DLSS5Enhance_AMD,
    "Dlss5AmdIntelSimple": RH_DLSS5Enhance_AMD,
    "Dlss5AmdImageEnhancer": RH_DLSS5Enhance_AMD,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RH_DLSS5Enhance_AMD": "RH DLSS5 Enhance (NR / Upscale) [AMD]",
    "RH_DLSS5FrameInterpolation_AMD": "RH DLSS5 Frame Interpolation [AMD]",
    "Dlss5AmdFsrSimple": "Dlss5 AMD FSR Simple (Legacy) [AMD]",
    "Dlss5AmdNvidiaSimple": "Dlss5 AMD Nvidia Simple (Legacy) [AMD]",
    "Dlss5AmdIntelSimple": "Dlss5 AMD Intel Simple (Legacy) [AMD]",
    "Dlss5AmdImageEnhancer": "Dlss5 AMD Image Enhancer (Legacy) [AMD]",
}

