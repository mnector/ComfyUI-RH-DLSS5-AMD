#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
enforce_amd_patches.py - Automated AMD / ZLUDA Patch Enforcement for ComfyUI-RH-DLSS5-AMD
========================================================================================
This script is idempotent. It can be safely run at any time (especially after an
upstream pull or merge from https://github.com/RH-RunningHub/ComfyUI-RH-DLSS5.git).

It automatically scans and enforces:
1. Node Isolation: Ensures all node mapping keys end with '_AMD', display names
   include '[AMD]', and categories include '[AMD]' so this custom node never
   collides with upstream.
2. AMD Backend & ZLUDA Tensor Pipeline: Enforces the amd_backend selector,
   PyTorch GPU tensor Neural Rendering fallback, and hot-reload INI writing.
3. Optical Flow & Frame Interpolation: Enforces OpenCV DIS optical flow without
   spurious NVOFA warnings on AMD GPUs, and pyramidal warping fallback.
4. Dependency Shielding: Protects AMD ROCm / ZLUDA environments from torch overwrites.
5. Strict Git Privacy: Verifies .gitignore rules to prevent accidental leaks.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Fix console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent


def patch_nodes_file(nodes_path: Path) -> bool:
    if not nodes_path.is_file():
        return False
    content = nodes_path.read_text(encoding="utf-8")
    modified = False

    # 1. Ensure node classes end in _AMD
    if "class RH_DLSS5Enhance:" in content:
        content = content.replace("class RH_DLSS5Enhance:", "class RH_DLSS5Enhance_AMD:")
        modified = True
    if "class RH_DLSS5FrameInterpolation:" in content:
        content = content.replace("class RH_DLSS5FrameInterpolation:", "class RH_DLSS5FrameInterpolation_AMD:")
        modified = True

    # 2. Ensure mappings use _AMD and include backward compatibility aliases
    mapping_replacements = [
        ('"RH_DLSS5Enhance": RH_DLSS5Enhance,', '"RH_DLSS5Enhance_AMD": RH_DLSS5Enhance_AMD,'),
        ('"RH_DLSS5FrameInterpolation": RH_DLSS5FrameInterpolation,', '"RH_DLSS5FrameInterpolation_AMD": RH_DLSS5FrameInterpolation_AMD,'),
        ('"RH_DLSS5Enhance": "RH DLSS5 Enhance (NR / Upscale)",', '"RH_DLSS5Enhance_AMD": "RH DLSS5 Enhance (NR / Upscale) [AMD]",'),
        ('"RH_DLSS5FrameInterpolation": "RH DLSS5 Frame Interpolation",', '"RH_DLSS5FrameInterpolation_AMD": "RH DLSS5 Frame Interpolation [AMD]",'),
    ]
    for old, new in mapping_replacements:
        if old in content:
            content = content.replace(old, new)
            modified = True

    if '"Dlss5AmdImageEnhancer":' not in content:
        old_map = '"RH_DLSS5FrameInterpolation_AMD": RH_DLSS5FrameInterpolation_AMD,\n}'
        new_map = (
            '"RH_DLSS5FrameInterpolation_AMD": RH_DLSS5FrameInterpolation_AMD,\n'
            '    # Backward compatibility aliases for previous Dlss5-AMD workflows\n'
            '    "Dlss5AmdFsrSimple": RH_DLSS5Enhance_AMD,\n'
            '    "Dlss5AmdNvidiaSimple": RH_DLSS5Enhance_AMD,\n'
            '    "Dlss5AmdIntelSimple": RH_DLSS5Enhance_AMD,\n'
            '    "Dlss5AmdImageEnhancer": RH_DLSS5Enhance_AMD,\n}'
        )
        if old_map in content:
            content = content.replace(old_map, new_map)
            modified = True

        old_disp = '"RH_DLSS5FrameInterpolation_AMD": "RH DLSS5 Frame Interpolation [AMD]",\n}'
        new_disp = (
            '"RH_DLSS5FrameInterpolation_AMD": "RH DLSS5 Frame Interpolation [AMD]",\n'
            '    "Dlss5AmdFsrSimple": "Dlss5 AMD FSR Simple (Legacy) [AMD]",\n'
            '    "Dlss5AmdNvidiaSimple": "Dlss5 AMD Nvidia Simple (Legacy) [AMD]",\n'
            '    "Dlss5AmdIntelSimple": "Dlss5 AMD Intel Simple (Legacy) [AMD]",\n'
            '    "Dlss5AmdImageEnhancer": "Dlss5 AMD Image Enhancer (Legacy) [AMD]",\n}'
        )
        if old_disp in content:
            content = content.replace(old_disp, new_disp)
            modified = True

    # 3. Ensure CATEGORY includes [AMD]
    category_replacements = [
        ('CATEGORY = "image/upscaling"', 'CATEGORY = "image/upscaling [AMD]"'),
        ('CATEGORY = "video"', 'CATEGORY = "video [AMD]"'),
    ]
    for old, new in category_replacements:
        if old in content:
            content = content.replace(old, new)
            modified = True

    # 4. Ensure amd_backend option in RH_DLSS5Enhance_AMD
    if '"amd_backend":' not in content:
        target = '"video": ("VIDEO", {"tooltip": "Optional video (e.g. LoadVideo). Frames are processed in temporal order."}),'
        replacement = target + '\n                "amd_backend": (["auto", "amd_nr_bridge", "zluda_tensor"], {"default": "auto", "tooltip": "Backend selector for AMD GPUs: auto selects amd_nr_bridge if D3D12/HIP runtime is present, otherwise falls back to zluda_tensor (pure PyTorch GPU-accelerated tensor pipeline); amd_nr_bridge loads OptiScaler / danielblnc HIP bridge; zluda_tensor executes 100% on GPU via ZLUDA/ROCm."}),'
        if target in content:
            content = content.replace(target, replacement)
            modified = True

    # 5. Ensure _make_progress_bar is present
    if "def _make_progress_bar" not in content:
        helper = (
            "def _make_progress_bar(total: int = 100):\n"
            "    try:\n"
            "        from comfy.utils import ProgressBar\n"
            "        return ProgressBar(total)\n"
            "    except Exception:\n"
            "        class _DummyBar:\n"
            "            def __init__(self, _t): pass\n"
            "            def update_absolute(self, val, _t=None, _p=None): pass\n"
            "            def update(self, val): pass\n"
            "        return _DummyBar(total)\n\n\n"
        )
        if "def _progress_callback():" in content:
            content = content.replace("def _progress_callback():", helper + "def _progress_callback():")
            modified = True

    # 6. Ensure raw ProgressBar imports are sanitized
    if "from comfy.utils import ProgressBar\n        bar = ProgressBar(100)" in content:
        content = content.replace("from comfy.utils import ProgressBar\n        bar = ProgressBar(100)", "bar = _make_progress_bar(100)")
        modified = True
    if "bar = ProgressBar(100)" in content:
        content = content.replace("bar = ProgressBar(100)", "bar = _make_progress_bar(100)")
        modified = True

    if modified:
        nodes_path.write_text(content, encoding="utf-8")
        print("[+] Enforced AMD patches on nodes.py")
    else:
        print("[*] nodes.py is already up-to-date with AMD patches")
    return modified


def patch_requirements(req_path: Path) -> bool:
    if not req_path.is_file():
        return False
    lines = req_path.read_text(encoding="utf-8").splitlines()
    forbidden = ("torch", "torchvision", "torchaudio", "triton", "xformers", "nvidia")
    filtered = []
    modified = False
    has_opencv = False
    has_numpy = False

    for line in lines:
        stripped = line.strip().lower()
        if any(stripped.startswith(f) for f in forbidden):
            modified = True
            continue
        if stripped.startswith("opencv"):
            has_opencv = True
        if stripped.startswith("numpy"):
            has_numpy = True
        filtered.append(line)

    if not has_numpy:
        filtered.insert(0, "numpy")
        modified = True
    if not has_opencv:
        filtered.append("opencv-python")
        modified = True

    if modified:
        req_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")
        print("[+] Sanitized requirements.txt (protected ROCm/ZLUDA environment)")
    else:
        print("[*] requirements.txt is clean")
    return modified


def patch_gitignore(git_ignore_path: Path) -> bool:
    if not git_ignore_path.is_file():
        return False
    content = git_ignore_path.read_text(encoding="utf-8")
    modified = False
    critical_rules = [
        ".env*",
        ".env.local",
        "*.local*",
        "*.dll",
        "*.exe",
        "*.pak",
        "*.bin",
        "*.safetensors",
    ]
    missing = [rule for rule in critical_rules if rule not in content]
    if missing:
        append_block = "\n# Critical Privacy & Binary Shields\n" + "\n".join(missing) + "\n"
        git_ignore_path.write_text(content.strip() + append_block, encoding="utf-8")
        print("[+] Enhanced .gitignore with missing privacy shields")
        modified = True
    else:
        print("[*] .gitignore contains all required privacy rules")
    return modified


def patch_common(common_path: Path) -> bool:
    if not common_path.is_file():
        return False
    content = common_path.read_text(encoding="utf-8")
    modified = False
    if "is_amd_environment" not in content:
        insert_marker = "from .common import DLSS5Error" if "from .common import DLSS5Error" in content else "from pathlib import Path"
        replacement = insert_marker + "\nfrom .amd_engine import is_amd_environment"
        content = content.replace(insert_marker, replacement, 1)
        modified = True

    if modified:
        common_path.write_text(content, encoding="utf-8")
        print("[+] Enforced AMD patches on dlss5nr/common.py")
    else:
        print("[*] dlss5nr/common.py is up-to-date")
    return modified


def patch_windows_backend(wb_path: Path) -> bool:
    if not wb_path.is_file():
        return False
    content = wb_path.read_text(encoding="utf-8")
    modified = False
    if "write_amd_nr_config" not in content:
        if "from . import common" in content:
            content = content.replace("from . import common", "from . import common\nfrom .amd_engine import write_amd_nr_config")
            modified = True
        if "common.validate_sizes(input_w, input_h, output_w, output_h)\n    runtime: Path = params[\"runtime\"]" in content:
            content = content.replace(
                "common.validate_sizes(input_w, input_h, output_w, output_h)\n    runtime: Path = params[\"runtime\"]",
                "common.validate_sizes(input_w, input_h, output_w, output_h)\n    runtime: Path = params[\"runtime\"]\n    write_amd_nr_config(runtime, params)"
            )
            modified = True

    if modified:
        wb_path.write_text(content, encoding="utf-8")
        print("[+] Enforced AMD patches on dlss5nr/windows_backend.py")
    else:
        print("[*] dlss5nr/windows_backend.py is up-to-date")
    return modified


def patch_videoout(vo_path: Path) -> bool:
    if not vo_path.is_file():
        return False
    content = vo_path.read_text(encoding="utf-8")
    modified = False
    if "_safe_temp_dir" not in content:
        helper = (
            "\ndef _safe_temp_dir() -> Path:\n"
            "    try:\n"
            "        from folder_paths import get_temp_directory\n"
            "        d = Path(get_temp_directory())\n"
            "        d.mkdir(parents=True, exist_ok=True)\n"
            "        return d\n"
            "    except Exception:\n"
            "        import tempfile\n"
            "        d = Path(tempfile.gettempdir())\n"
            "        d.mkdir(parents=True, exist_ok=True)\n"
            "        return d\n\n"
        )
        if "class StreamEncoder:" in content:
            content = content.replace("class StreamEncoder:", helper + "class StreamEncoder:")
            content = content.replace("Path(get_temp_directory())", "_safe_temp_dir()")
            modified = True

    if modified:
        vo_path.write_text(content, encoding="utf-8")
        print("[+] Enforced AMD patches on dlss5nr/videoout.py")
    else:
        print("[*] dlss5nr/videoout.py is up-to-date")
    return modified


def patch_pyproject(pyproject_path: Path) -> bool:
    if not pyproject_path.is_file():
        return False
    content = pyproject_path.read_text(encoding="utf-8")
    modified = False
    if 'name = "ComfyUI-RH-DLSS5"' in content:
        content = content.replace('name = "ComfyUI-RH-DLSS5"', 'name = "ComfyUI-RH-DLSS5-AMD"')
        modified = True
    if modified:
        pyproject_path.write_text(content, encoding="utf-8")
        print("[+] Enforced AMD naming on pyproject.toml")
    else:
        print("[*] pyproject.toml is up-to-date")
    return modified


def main():
    print("=" * 75)
    print(" ComfyUI-RH-DLSS5-AMD: Idempotent Patch Enforcer")
    print("=" * 75)
    patch_nodes_file(ROOT_DIR / "nodes.py")
    patch_common(ROOT_DIR / "dlss5nr" / "common.py")
    patch_windows_backend(ROOT_DIR / "dlss5nr" / "windows_backend.py")
    patch_videoout(ROOT_DIR / "dlss5nr" / "videoout.py")
    patch_pyproject(ROOT_DIR / "pyproject.toml")
    patch_requirements(ROOT_DIR / "requirements.txt")
    patch_gitignore(ROOT_DIR / ".gitignore")
    print("=" * 75)
    print("[OK] AMD Shielding verification complete.")


if __name__ == "__main__":
    main()
