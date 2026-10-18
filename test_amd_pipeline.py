import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent

import importlib.util
spec = importlib.util.spec_from_file_location(
    "rh_dlss5_amd", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
pkg = importlib.util.module_from_spec(spec)
sys.modules["rh_dlss5_amd"] = pkg
spec.loader.exec_module(pkg)

def test_pipeline():
    print("=" * 70)
    print(" ComfyUI-RH-DLSS5-AMD : End-to-End AMD Verification Test")
    print("=" * 70)

    from rh_dlss5_amd.nodes import RH_DLSS5Enhance_AMD, RH_DLSS5FrameInterpolation_AMD
    from rh_dlss5_amd.dlss5nr.amd_engine import is_amd_environment

    print(f"[*] PyTorch Version : {torch.__version__}")
    print(f"[*] CUDA Available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[*] Device Name     : {torch.cuda.get_device_name(0)}")
    print(f"[*] Is AMD Detected : {is_amd_environment()}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Test RH_DLSS5Enhance_AMD with zluda_tensor
    print("\n--- [Test 1] Testing RH_DLSS5Enhance_AMD (amd_backend='zluda_tensor') ---")
    enhancer = RH_DLSS5Enhance_AMD()
    # Batch of 2 frames 256x256 RGB float32 in [0, 1]
    input_img = torch.rand((2, 256, 256, 3), dtype=torch.float32, device=device)
    out_img, out_vid, note = enhancer.enhance(
        image=input_img,
        amd_backend="zluda_tensor",
        upscaling_mode="2x (Performance)",
        style="cinematic",
        intensity=1.0,
        structure=1.5,
    )
    assert out_img is not None, "zluda_tensor returned None"
    assert out_img.shape == (2, 512, 512, 3), f"Unexpected shape {out_img.shape}"
    mean_val = float(out_img.mean())
    assert mean_val > 0.05, f"Output is too dark/black: mean={mean_val}"
    print(f"[OK] zluda_tensor succeeded: shape={out_img.shape}, mean={mean_val:.4f}")
    print(f"     Status note: {note}")

    # 2. Test RH_DLSS5FrameInterpolation_AMD (OpenCV DIS + PyTorch warping)
    print("\n--- [Test 2] Testing RH_DLSS5FrameInterpolation_AMD (motion='auto' -> DIS) ---")
    interpolator = RH_DLSS5FrameInterpolation_AMD()
    out_interp, out_vid, note = interpolator.interpolate(
        image=input_img,
        output_fps="2x",
        motion="auto",
        scene_change_threshold=0.24,
    )
    assert out_interp is not None, "Frame interpolation returned None"
    assert out_interp.shape[0] == 3, f"Expected 3 frames for 2x on 2 frames, got {out_interp.shape[0]}"
    interp_mean = float(out_interp.mean())
    assert interp_mean > 0.05, f"Interpolated output is too dark: mean={interp_mean}"
    print(f"[OK] Frame interpolation succeeded: shape={out_interp.shape}, mean={interp_mean:.4f}")
    print(f"     Status note: {note}")

    # 3. Test RH_DLSS5Enhance_AMD with amd_nr_bridge (OptiScaler D3D12/HIP)
    print("\n--- [Test 3] Testing RH_DLSS5Enhance_AMD (amd_backend='amd_nr_bridge') ---")
    try:
        # Create a real recognizable image (e.g. gradient/pattern) 512x512
        test_single = torch.zeros((1, 512, 512, 3), dtype=torch.float32, device=device)
        for c in range(3):
            test_single[0, :, :, c] = (torch.linspace(0.2, 0.8, 512).unsqueeze(0) + c * 0.1).clamp(0, 1)

        # Test single frame
        out_bridge, out_vid, note = enhancer.enhance(
            image=test_single,
            amd_backend="amd_nr_bridge",
            upscaling_mode="1x (DLAA / native)",
            style="default",
            intensity=1.0,
            structure=1.5,
        )
        assert out_bridge is not None, "amd_nr_bridge returned None"
        b_mean = float(out_bridge.mean())
        print(f"[OK] amd_nr_bridge single frame succeeded: shape={out_bridge.shape}, mean={b_mean:.4f}")
        print(f"     Status note: {note}")
        assert b_mean > 0.01, f"Bridge output is black (mean={b_mean})"

        # Test multi-frame streaming (3 frames)
        print("\n--- [Test 4] Testing RH_DLSS5Enhance_AMD multi-frame streaming (3 frames) ---")
        test_seq = torch.rand((3, 256, 256, 3), dtype=torch.float32, device=device)
        out_seq, _, note_seq = enhancer.enhance(
            image=test_seq,
            amd_backend="amd_nr_bridge",
            upscaling_mode="1x (DLAA / native)",
            style="cinematic",
            intensity=1.0,
            structure=1.2,
        )
        assert out_seq is not None, "amd_nr_bridge multi-frame returned None"
        assert out_seq.shape == (3, 256, 256, 3), f"Unexpected multi-frame shape: {out_seq.shape}"
        seq_mean = float(out_seq.mean())
        print(f"[OK] amd_nr_bridge multi-frame succeeded: shape={out_seq.shape}, mean={seq_mean:.4f}")
        print(f"     Status note: {note_seq}")
        assert seq_mean > 0.01, f"Multi-frame output is black (mean={seq_mean})"
    except Exception as e:
        print(f"[!] amd_nr_bridge test error: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("\n" + "=" * 70)
    print(" ALL TESTS PASSED SUCCESSFULLY ON AMD RADEON GPU! ")
    print("=" * 70)

if __name__ == "__main__":
    test_pipeline()
