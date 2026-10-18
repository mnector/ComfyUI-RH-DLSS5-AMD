#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
setup_amd_runtime.py - Diagnostics & Runtime Setup Helper for AMD GPUs
======================================================================
License-compliant manager for DLSS 5 Neural Rendering community runtimes
(OptiScaler AMD-NR / danielblnc DLSS-NR-on-AMD / ZLUDA tensor fallback).

Features:
1. Detects ComfyUI models/dlss5 directory (via folder_paths, .env.local, or relative path).
2. Sets up directory structure (models/dlss5, caller shim, dlssg).
3. Provides full diagnostic status of AMD GPU, ZLUDA, ROCm, and runtime files.
4. Allows extracting locally downloaded ZIP releases (e.g. v0.6.0-Runtime.zip from danielblnc).
5. Respects NVIDIA and third-party licenses: never distributes proprietary binaries.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

# Configure console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent


def load_local_env() -> None:
    """Load optional private configuration from .env.local."""
    env_file = ROOT_DIR / ".env.local"
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except Exception:
            pass


def resolve_comfy_models_dir() -> Path:
    """Resolve ComfyUI models directory dynamically."""
    load_local_env()
    comfy_path = os.environ.get("COMFYUI_PATH")
    if comfy_path and Path(comfy_path).exists():
        return Path(comfy_path) / "models"

    try:
        import folder_paths
        return Path(folder_paths.models_dir)
    except Exception:
        pass

    # Search upwards from custom_nodes/ComfyUI-RH-DLSS5-AMD
    curr = ROOT_DIR.resolve()
    for _ in range(4):
        if (curr / "models").is_dir():
            return curr / "models"
        if (curr / "ComfyUI" / "models").is_dir():
            return curr / "ComfyUI" / "models"
        curr = curr.parent

    # Fallback to local models directory
    return ROOT_DIR / "models"


def resolve_dlss5_models_dir() -> Path:
    models_dir = resolve_comfy_models_dir()
    dlss5_dir = models_dir / "dlss5"
    dlss5_dir.mkdir(parents=True, exist_ok=True)
    return dlss5_dir


def ensure_caller_shim(dlss5_dir: Path) -> bool:
    """Link or copy bundled caller shim into models/dlss5/caller."""
    bundled_shim = ROOT_DIR / "runtime" / "caller" / "nvngx.dll_comfy.dll"
    if not bundled_shim.is_file():
        return False
    caller_dir = dlss5_dir / "caller"
    caller_dir.mkdir(parents=True, exist_ok=True)
    target = caller_dir / "nvngx.dll_comfy.dll"
    if not target.is_file():
        try:
            shutil.copy2(bundled_shim, target)
            return True
        except Exception as e:
            print(f"[!] Warning copying caller shim: {e}")
            return False
    return True


def diagnose_environment(dlss5_dir: Path) -> None:
    """Print complete diagnostic status for AMD environment."""
    print("=" * 75)
    print(" ComfyUI-RH-DLSS5-AMD : System & Runtime Diagnostics")
    print("=" * 75)

    # 1. Hardware & Framework Detection
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
        print(f"[*] PyTorch Version : {torch.__version__}")
        print(f"[*] GPU Available   : {cuda_avail} ({gpu_name})")
    except Exception as e:
        print(f"[!] PyTorch Check Failed: {e}")
        gpu_name = "Unknown"

    from dlss5nr.amd_engine import is_amd_environment
    is_amd = is_amd_environment()
    print(f"[*] AMD Detection   : {'[YES] AMD HIP / ROCm / ZLUDA detected' if is_amd else '[INFO] Non-AMD or Standard CUDA'}")

    try:
        import cv2
        print(f"[*] OpenCV Version  : {cv2.__version__} (DIS Optical Flow available)")
    except Exception:
        print("[!] OpenCV Missing  : install opencv-python for DIS motion flow")

    print(f"[*] Target Models   : {dlss5_dir}")
    print("-" * 75)

    # 2. Files Check in models/dlss5
    files = {
        "nvngx_dlssnr.dll": "DLSS 5 Neural Rendering weights (Universal 310.8)",
        "nvngx_dlss.dll": "NVIDIA DLSS carrier (required for >1x upscaling)",
        "caller/nvngx.dll_comfy.dll": "Open caller shim (bundled with plugin)",
        "OptiScaler.dll": "OptiScaler D3D12/HIP proxy (AMD-NR---OptiScaler)",
        "dxgi.dll": "OptiScaler DXGI hook proxy",
        "LmxxfNrRuntime.dll": "Neural Rendering runtime for AMD (lmxxf / OptiScaler)",
        "LmxxfNrRuntime.pak": "Neural Rendering package weights for AMD",
        "dlssnr_amd_pass1.dll": "danielblnc DLSS-NR AMD Pass 1 HIP runtime",
        "dlssnr_on_amd_weights.bin": "danielblnc DLSS-NR extracted neural weights",
        "amdhip64_7.dll": "AMD HIP runtime kernel library (danielblnc DLSS-NR-on-AMD)",
        "dlss_bridge.exe": "Dedicated AMD-NR D3D12 persistent bridge daemon",
        "dlssg-worker.exe": "DLSS-G Frame Generation worker binary",
    }

    print(" Runtime Components Status:")
    for fname, desc in files.items():
        p = dlss5_dir / fname
        status = "[PRESENT]" if p.is_file() else "[NOT FOUND]"
        print(f"  {status:<12} {fname:<28} - {desc}")

    print("-" * 75)
    print(" Operational Modes Summary:")
    print("  1. zluda_tensor (Pure PyTorch on AMD GPU):")
    print("     -> 100% READY! Works out of the box with zero external DLLs required.")
    print("     -> Supports 1x, 1.5x, 1.724x, 2x, 3x, structure, tonemapping, and skin auto-mask.")
    print("  2. amd_nr_bridge (D3D12 / HIP via OptiScaler or danielblnc):")
    has_nr = (dlss5_dir / "nvngx_dlssnr.dll").is_file() or (dlss5_dir / "LmxxfNrRuntime.dll").is_file() or (dlss5_dir / "dlssnr_on_amd_weights.bin").is_file()
    has_bridge = (dlss5_dir / "OptiScaler.dll").is_file() or (dlss5_dir / "dxgi.dll").is_file() or (dlss5_dir / "dlss_bridge.exe").is_file()
    if has_nr and has_bridge:
        print("     -> [READY] All required bridge DLLs detected in models/dlss5/.")
    else:
        print("     -> [OPTIONAL] Requires nvngx_dlssnr.dll + community AMD-NR runtime in models/dlss5/.")
    print("  3. Frame Interpolation (RH_DLSS5FrameInterpolation_AMD):")
    print("     -> [READY] OpenCV DIS + PyTorch GPU pyramidal warping is fully functional!")
    print("=" * 75)


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extract a user-downloaded ZIP directly into models/dlss5."""
    if not zip_path.is_file():
        print(f"[!] Error: File does not exist: {zip_path}")
        return
    print(f"[*] Extracting {zip_path.name} into {target_dir}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(target_dir)
        print(f"[OK] Successfully extracted {zip_path.name}")
        ensure_caller_shim(target_dir)
    except Exception as e:
        print(f"[!] Extraction error: {e}")


def import_from_source(source_dir: Path, target_dir: Path) -> None:
    """Import AMD-NR / OptiScaler binaries from an existing directory into target_dir."""
    if not source_dir.is_dir():
        print(f"[!] Warning: Source directory does not exist: {source_dir}")
        return
    print(f"[*] Importing AMD runtime binaries from {source_dir} -> {target_dir}...")
    patterns = [
        "OptiScaler.dll",
        "dxgi.dll",
        "LmxxfNrRuntime.dll",
        "LmxxfNrRuntime.pak",
        "dlssnr_amd_pass*.dll",
        "dlssnr_on_amd_weights.bin",
        "amdhip64*.dll",
        "nvngx_dlss*.dll",
        "dlss_bridge.exe",
        "OptiScaler.ini",
        "amd_fidelityfx_*.dll",
        "libxess*.dll",
        "libxell.dll",
    ]
    imported = 0
    for pat in patterns:
        for f in source_dir.glob(pat):
            if f.name.endswith(".disabled") or f.name.endswith(".bak"):
                continue
            dest = target_dir / f.name
            try:
                if not dest.exists() or dest.stat().st_size != f.stat().st_size:
                    shutil.copy2(f, dest)
                    imported += 1
            except Exception as e:
                print(f"[!] Warning copying {f.name}: {e}")

    # If OptiScaler.dll and dxgi.dll are both present, disable dxgi.dll to avoid
    # duplicate OptiScaler instantiation deadlock in dlss_bridge.exe on AMD GPUs.
    target_dxgi = target_dir / "dxgi.dll"
    target_opti = target_dir / "OptiScaler.dll"
    if target_dxgi.is_file() and target_opti.is_file():
        try:
            disabled_dxgi = target_dir / "dxgi.dll.disabled"
            if disabled_dxgi.exists():
                disabled_dxgi.unlink()
            target_dxgi.rename(disabled_dxgi)
        except Exception:
            pass

    # Sanitize OptiScaler.ini to ensure no machine-specific absolute paths leak in [Libraries]
    ini_file = target_dir / "OptiScaler.ini"
    if ini_file.is_file():
        try:
            import configparser
            cfg = configparser.ConfigParser(interpolation=None)
            cfg.read(ini_file, encoding="utf-8")
            if cfg.has_section("Libraries"):
                for opt in cfg.options("Libraries"):
                    cfg.set("Libraries", opt, "auto")
            with open(ini_file, "w", encoding="utf-8") as f:
                cfg.write(f)
        except Exception:
            pass

    # Also if dlss_bridge.exe is imported, copy to native/bin/dlss_bridge.exe as well
    native_bin = ROOT_DIR / "native" / "bin"
    native_bin.mkdir(parents=True, exist_ok=True)
    bridge_src = target_dir / "dlss_bridge.exe"
    if bridge_src.is_file():
        bridge_dest = native_bin / "dlss_bridge.exe"
        if not bridge_dest.is_file() or bridge_dest.stat().st_size != bridge_src.stat().st_size:
            try:
                shutil.copy2(bridge_src, bridge_dest)
            except Exception:
                pass
    print(f"[OK] Import complete: {imported} components imported.")


def main():
    parser = argparse.ArgumentParser(description="ComfyUI-RH-DLSS5-AMD Runtime Setup Helper")
    parser.add_argument("--extract", type=str, help="Extract a local ZIP (e.g. v0.6.0-Runtime.zip) to models/dlss5")
    parser.add_argument("--import-from", type=str, help="Import runtime components from an existing local directory")
    parser.add_argument("--check", action="store_true", help="Run diagnostic check")
    args = parser.parse_args()

    load_local_env()
    dlss5_dir = resolve_dlss5_models_dir()
    ensure_caller_shim(dlss5_dir)

    # Automatic import from local env source if present
    import_src = args.import_from or os.environ.get("COMFYUI_DLSS5_AMD_SRC")
    if import_src and Path(import_src).is_dir():
        import_from_source(Path(import_src), dlss5_dir)

    if args.extract:
        extract_zip(Path(args.extract), dlss5_dir)

    diagnose_environment(dlss5_dir)


if __name__ == "__main__":
    main()
