<div align="center">

# ⚡ ComfyUI-RH-DLSS5-AMD
### DLSS 5 Neural Rendering (Upscale & Enhance) + Temporal Frame Generation for AMD Radeon GPUs on Windows

[![AMD ROCm & ZLUDA](https://img.shields.io/badge/AMD%20GPU-ROCm%20%7C%20ZLUDA-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/Speedstu/CUDA-for-AMD-Windows)
[![DLSS 5 Neural Rendering](https://img.shields.io/badge/DLSS%205-Neural%20Rendering%20(Feature%2018)-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://github.com/RH-RunningHub/ComfyUI-RH-DLSS5)
[![Community Runtime AMD-NR](https://img.shields.io/badge/Community%20Runtime-AMD--NR%20%7C%20OptiScaler-FF8000?style=for-the-badge&logo=open-source-initiative&logoColor=white)](https://github.com/3zwr1/AMD-NR---OptiScaler)
[![ComfyUI Node](https://img.shields.io/badge/ComfyUI-Custom%20Node%20[AMD]-0A84FF?style=for-the-badge&logo=comfyui&logoColor=white)](https://github.com/comfyanonymous/ComfyUI)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OS Windows](https://img.shields.io/badge/OS-Windows%2011%2064--bit-0078D4?style=for-the-badge&logo=windows&logoColor=white)](https://microsoft.com/windows)
[![Donate via PayPal](https://img.shields.io/badge/Donate-PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/mnecstream)
[![License MIT](https://img.shields.io/badge/License-MIT-5856D6?style=for-the-badge)](LICENSE)

<p align="center">
  <b>The first production-ready, hybrid architecture bringing NVIDIA DLSS 5 Neural Rendering (Feature 18) and high-framerate temporal Frame Generation to AMD Radeon graphics cards on Windows via ZLUDA, AMD HIP, and OpenCV DIS.</b>
</p>

[Why This Fork Exists](#-the-technical-story-why-does-this-fork-even-exist) •
[Architecture](#-pipeline-architecture) •
[Auto-Updater](#-auto-update-system--amd-shielding) •
[Setup & Runtimes](#-runtime-setup--installation) •
[Workflow](#-ready-to-use-workflow) •
[Troubleshooting](#-troubleshooting--pro-tips) •
[Support](#-support-this-project--buy-me-a-coffee) •
[Credits & Licenses](#-licenses--shoutouts)

---

</div>

## 📖 The Technical Story: Why does this fork even exist?

NVIDIA's **DLSS 5 Neural Rendering (Feature 18)** is a massive leap forward in image reconstruction, micro-structure preservation, and real-time upscaling. The upstream project [`RH-RunningHub/ComfyUI-RH-DLSS5`](https://github.com/RH-RunningHub/ComfyUI-RH-DLSS5) did amazing work building low-latency D3D12 streaming bridges and frame generation pipelines for ComfyUI.

However, upstream was built **exclusively for NVIDIA RTX hardware**:
1. **Missing DriverStore NGX Core:** On AMD GPUs, NVIDIA's proprietary `_nvngx.dll` does not exist in `DriverStore\FileRepository`. The native C++ bridge aborted on startup with `Could not find _nvngx.dll`.
2. **Missing Optical Flow Hardware (NVOFA):** Upstream frame generation depends on NVIDIA's proprietary hardware accelerator (`nvof`). On AMD GPUs, attempting to interpolate frames threw fatal errors due to missing NVOFA contexts.
3. **Hardware Vendor Lockout:** The native D3D12 device discovery rejected any adapter whose `VendorId != 0x10DE` (NVIDIA), completely locking out AMD Radeon hardware (`0x1002`).

### How this fork solves every hurdle:

This fork bridges the gap by uniting modern community D3D12/HIP translation runtimes with a high-performance pure-PyTorch GPU engine ("Best of Both Worlds"):

- 🚀 **`zluda_tensor` Mode ("Best of Both Worlds" Pure GPU Engine):**
  - **100% In-VRAM GPU Execution:** Accelerated directly by your AMD Radeon card via [`Speedstu/CUDA-for-AMD-Windows`](https://github.com/Speedstu/CUDA-for-AMD-Windows) (ZLUDA + AMD HIP SDK / ROCm).
  - **Zero Black Screens & Zero Window Popups:** Operates entirely inside the ComfyUI PyTorch process with zero external window creation, zero DirectX swapchain flashing, and zero OS display deadlocks.
  - **FidelityFX CAS (Contrast Adaptive Sharpening):** GPU tensor-accelerated CAS filter recovers micro-detail and edge contrast with user-adjustable strength (`sharpening`).
  - **Full AMD-NR Parameter Suite (1:1 with AMDNR v0.3.5.1 GUI):**
    - **Tuning Styles (`nr_style`):** `Default`, `Cinematic` (filmic contrast, rich shadows), `Crisp` (maximum edge definition & CAS clarity), `Natural` (faithful micro-textures), `Vivid` (punchy saturation/highlights), or `Custom`.
    - **Residual Detail Transfer:** Micro-structure frequency reconstruction (`residual_strength` / `TransferStrength`) with ratio clamping (`residual_limit` / `MaxRatio`) to prevent ringing.
    - **Temporal Stability:** Inter-frame history accumulation (`temporal_stability` / `AmdTemporalStability`) eliminates flicker.
    - **Scaling Modes:** `1x (Native / DLAA)`, `1.5x (Quality)`, `1.724x (Balanced)`, `2x (Performance)`, `3x (Ultra Performance)`, and bucketed targets (`1K`, `2K`, `4K`, `8K`).
    - **Self-Contained Image/Video Pipeline:** Standard `image` and `video` inputs only. Auxiliary depth, surface normals, albedo, and optical flow are all generated internally on-the-fly without requiring external nodes!
- 🎮 **`amd_nr_bridge` Mode (Offscreen Hidden D3D12/HIP Bridge):**
  - Enables D3D12/HIP translation using community shims ([`3zwr1/AMD-NR---OptiScaler`](https://github.com/3zwr1/AMD-NR---OptiScaler), [`danielblnc/DLSS-NR-on-AMD`](https://github.com/danielblnc/DLSS-NR-on-AMD), and `LmxxfNrRuntime`).
  - **Zero Black Screen Flash:** Built with an offscreen hidden tool window (`WS_EX_TOOLWINDOW` at coordinates `-32000, -32000`), completely eliminating desktop blackouts and fullscreen capture popups.
  - **Hot-Reload Parameter Sync:** Automatically writes real-time configuration files to `dlssnr_on_amd.ini` and `OptiScaler.ini` (`[DlssNr]` and `[AmdLook]`).
- ⚡ **Temporal Frame Generation without NVOFA (`RH_DLSS5FrameInterpolation_AMD`):**
  - Seamlessly replaces NVIDIA's hardware optical flow with **OpenCV DIS (Dense Inverse Search)** optical flow.
  - Generates bidirectional motion fields ($F_{0 \to 1}$ and $F_{1 \to 0}$) with occlusion masking and executes sub-pixel backward warping directly on AMD GPU VRAM using PyTorch `grid_sample`.
  - Supports 2x, 3x, 4x, 6x multipliers or arbitrary target frame rates (`output_fps`), scene change cut detection (`scene_change_threshold`), and bounded-memory FFmpeg streaming with audio preservation.
- 🛡️ **Zero-Collision Isolation (`_AMD` / `[AMD]`):**
  - All class mappings (`RH_DLSS5Enhance_AMD`, `RH_DLSS5FrameInterpolation_AMD`) and UI display titles include `[AMD]` to allow conflict-free coexistence alongside upstream nodes.
- 🔄 **Self-Maintaining Upstream Updater & Auto-Rollback:**
  - Update effortlessly with `update_amd_node.bat`. If an upstream change introduces syntax errors or breaks compatibility, the updater triggers an automatic rollback (`git reset --hard`) to ensure your installation never breaks.

---

## 📊 Quick Comparison

| Feature | Upstream (ComfyUI-RH-DLSS5) | **ComfyUI-RH-DLSS5-AMD (This Fork)** |
| :--- | :---: | :---: |
| **NVIDIA RTX Support** | ✅ Native D3D12 | ✅ Fully Preserved |
| **AMD Radeon Support (Windows)** | ❌ *Hardware Lockout* | ✅ **Full Support (ZLUDA / ROCm / AMD HIP)** |
| **DLSS 5 Neural Rendering Bridge** | NVIDIA RTX Only | ✅ **D3D12/HIP Proxy (`OptiScaler` / `danielblnc`)** |
| **Zero-DLL Fallback Engine** | ❌ None (Aborts) | ✅ **`zluda_tensor` Pure PyTorch GPU Pipeline** |
| **Optical Flow / Frame Generation** | ⚠️ *NVOFA Only* | ✅ **OpenCV DIS + PyTorch GPU Warping** |
| **Node Naming & Workflow Safety** | Default | ✅ **Isolated `_AMD` & `[AMD]` Categories** |
| **Auto-Sync & Auto-Rollback** | ❌ Manual | ✅ **`update_amd_node.bat` Idempotent Engine** |

---

## 🏗️ Pipeline Architecture

```mermaid
flowchart TD
    classDef inputStyle fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#fff;
    classDef enhanceStyle fill:#1a365d,stroke:#2b6cb0,stroke-width:2px,color:#fff;
    classDef amdBridge fill:#742a2a,stroke:#c53030,stroke-width:2px,color:#fff;
    classDef zludaStyle fill:#7b341e,stroke:#dd6b20,stroke-width:2px,color:#fff;
    classDef fgStyle fill:#22543d,stroke:#38a169,stroke-width:2px,color:#fff;
    classDef outStyle fill:#234e52,stroke:#319795,stroke-width:2px,color:#fff;

    Input["🎬 Input Image / Video Frames"]:::inputStyle --> Router{"RH_DLSS5Enhance_AMD<br/>Backend Selector"}:::enhanceStyle

    Router -- "amd_nr_bridge<br/>(D3D12 / HIP Proxy)" --> Bridge["AMD-NR / OptiScaler Runtime<br/>(OptiScaler.dll / LmxxfNrRuntime.dll)"]:::amdBridge
    Bridge --> HotReload["Hot-Reload INI Generator<br/>(dlssnr_on_amd.ini / OptiScaler.ini)"]:::amdBridge
    HotReload --> D3DDevice["D3D12 Device (VendorId 0x1002 AMD)"]:::amdBridge
    D3DDevice --> BridgeOut["Neural Enhanced Frames"]:::outStyle

    Router -- "zluda_tensor<br/>(Pure GPU PyTorch)" --> ZludaPipe["ZLUDA / ROCm Tensor Engine"]:::zludaStyle
    ZludaPipe --> Scale["Neural Scale (1x, 1.5x, 1.724x, 2x, 3x, 4K/8K)"]:::zludaStyle
    Scale --> Color["Linear Tone Curve & Style (natural/cinematic/hdr)"]:::zludaStyle
    Color --> Skin["YCbCr / HSV Skin Masking & Detail Blend"]:::zludaStyle
    Skin --> ZludaOut["Tensor Enhanced Frames"]:::outStyle

    BridgeOut & ZludaOut --> FG{"RH_DLSS5FrameInterpolation_AMD<br/>Motion Mode"}:::fgStyle

    FG -- "NVIDIA NVOFA" --> NvofErr["NVOFA Unsupported on AMD<br/>(Auto-redirects to DIS)"]:::amdBridge
    FG -- "Auto / DIS (OpenCV)" --> DIS["OpenCV DIS Optical Flow<br/>Bidirectional Vector Fields (F0->1, F1->0)"]:::fgStyle
    DIS --> CutDetect["Temporal Scene Cut Detection<br/>(scene_change_threshold)"]:::fgStyle
    CutDetect --> Warp["PyTorch GPU Backward Warping<br/>(F.grid_sample on VRAM)"]:::fgStyle
    Warp --> FinalVid["📦 High-FPS Smooth Video Output<br/>(FFmpeg Streaming + Preserved Audio)"]:::outStyle
```

---

## 🔄 Auto-Update System & AMD Shielding

Keeping up with upstream changes shouldn't mean re-applying your AMD patches by hand every time. This fork includes an intelligent, idempotent synchronization and shielding system:

```mermaid
sequenceDiagram
    autonumber
    actor Dev as You
    participant Script as update_amd_node.bat
    participant Git as Git Engine
    participant Patcher as enforce_amd_patches.py
    participant Venv as Python (ComfyUI venv)

    Dev->>Script: Run update_amd_node.bat
    Script->>Git: Snapshots LAST_GOOD_COMMIT & tags backup-pre-sync
    Script->>Git: git fetch upstream && git merge upstream/main
    
    alt Merge Conflict?
        Git-->>Script: Conflict detected
        Script->>Git: git merge --abort && git reset --hard LAST_GOOD_COMMIT
        Script-->>Dev: ⚠️ Auto-Rollback triggered: Repository restored to working state
    else Clean Merge
        Script->>Patcher: Runs enforce_amd_patches.py
        Patcher->>Patcher: 1. Preserves _AMD class names & [AMD] tags
        Patcher->>Patcher: 2. Protects requirements.txt (safeguards ROCm/ZLUDA torch)
        Patcher->>Patcher: 3. Preserves D3D12 AMD VendorId & OpenCV DIS routing
        Script->>Venv: Runs Pre-flight Smoke Test on GPU
        alt Smoke Test Fails?
            Venv-->>Script: Import or execution failure
            Script->>Git: git reset --hard LAST_GOOD_COMMIT
            Script-->>Dev: ⚠️ Auto-Rollback triggered: Your environment stays pristine
        else Smoke Test Passes
            Script->>Git: git commit -m "Auto-sync with upstream + enforce AMD ZLUDA patches"
            Script-->>Dev: ✅ Synchronized and verified successfully!
        end
    end
```

### Updating in One Step:
Simply double-click:
```bat
update_amd_node.bat
```
Or execute from PowerShell / command prompt:
```powershell
<COMFYUI_PYTHON> update_from_upstream.py
```

---

## 📥 Runtime Setup & Installation

### 1. Requirements
- **Windows 11 (64-bit)**
- **AMD Radeon GPU:** RX 6000, 7000, 8000, 9000 series, or Radeon AI PRO (e.g. R9700).
- **ZLUDA Layer:** Installed and active via [`Speedstu/CUDA-for-AMD-Windows`](https://github.com/Speedstu/CUDA-for-AMD-Windows).
- **ComfyUI Virtual Environment:** With PyTorch built for ROCm/ZLUDA.

> [!CAUTION]
> **Do not run `pip install torch`!** This repository protects your existing ROCm/ZLUDA PyTorch build. Only install auxiliary packages:
> ```powershell
> <COMFYUI_PYTHON> -m pip install numpy opencv-python
> ```

### 2. Clone into ComfyUI Custom Nodes
In your ComfyUI custom nodes directory:
```bash
cd <COMFYUI_ROOT>/custom_nodes
git clone https://github.com/mnector/ComfyUI-RH-DLSS5-AMD.git
```

### 3. Understanding Backend Runtimes (License Compliance)

> [!IMPORTANT]
> **Licensing Notice:** Neither NVIDIA's proprietary DLLs (`nvngx_dlssnr.dll`) nor community binaries (`danielblnc/DLSS-NR-on-AMD`) permit redistribution in third-party git repositories.
> 
> You have two simple ways to run this custom node:
> 
> 1. **Zero External Files (`zluda_tensor` mode):**  
>    Works **immediately** after cloning! The pure PyTorch tensor engine runs directly on your AMD GPU via ZLUDA with zero setup required.
> 
> 2. **Hardware D3D12/HIP Bridge (`amd_nr_bridge` mode):**  
>    Download the runtime files directly from the original creators:
>    - Download the runtime release from [danielblnc/DLSS-NR-on-AMD](https://github.com/danielblnc/DLSS-NR-on-AMD) or [3zwr1/AMD-NR---OptiScaler](https://github.com/3zwr1/AMD-NR---OptiScaler).
>    - Place the extracted files into `<COMFYUI_ROOT>/models/dlss5/`.

### 4. Automated Runtime Diagnostic
Run our diagnostic helper at any time to inspect your runtime folder:
```powershell
<COMFYUI_PYTHON> setup_amd_runtime.py
```
It will display the status of each component and guide you if anything is missing:
```
======================================================================
 ComfyUI-RH-DLSS5-AMD: Runtime Setup & Diagnostics
======================================================================
[*] ComfyUI Models Target: <COMFYUI_ROOT>/models/dlss5
[*] Architecture: AMD Radeon GPU on Windows (ZLUDA / ROCm)

[+] Runtime folder exists and is ready for use.
[i] zluda_tensor fallback: 100% functional out-of-the-box!
======================================================================
```

---

## ⚙️ Node Parameters Reference (`RH_DLSS5Enhance_AMD`)

The parameters in this node match the official **AMD-NR GUI (v0.3.5.1)** 1:1, giving you direct control over the Neural Rendering pipeline:

> [!NOTE]
> **No Auxiliary Map Inputs Needed!**  
> Unlike game engines that require depth buffers, G-buffers, and surface normals, `RH_DLSS5Enhance_AMD` is designed as a standalone media enhancer (just like upstream `ComfyUI-RH-DLSS5`). Connect only your **`image`** or **`video`**. The node automatically synthesizes neutral depth planes, forward normals, albedo passthroughs, and OpenCV DIS optical flow internally.

| Category | Parameter | Type / Values | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Media** | `image` | `IMAGE` (optional) | *None* | Input image batch. Frames are processed in temporal sequence. |
| **Media** | `video` | `VIDEO` (optional) | *None* | Input video stream. Processed chunk-by-chunk with audio preservation. |
| **AMD-NR Core** | `enable_nr` | `BOOLEAN` | `True` | Master toggle to enable or bypass DLSS 5 Neural Rendering. |
| **AMD-NR Core** | `neural_runtime` | `danielblnc 0.5.1` \| `lmxxf` | `danielblnc 0.5.1` | AMD neural engine (danielblnc native HIP kernels or lmxxf runtime). |
| **AMD-NR Core** | `upscaling_mode` | `1x` .. `3x`, `1K`, `2K`, `4K`, `8K` | `1x (Native / DLAA)` | Upscaling factor or auto-fit resolution target. |
| **AMD-NR Core** | `nr_style` | `Full Model Look`, `NVIDIA look`, `AMDNR soft`, `Cinematic`, `Crisp`, `Natural`, `Vivid`, `Custom` | `Full Model Look` | AMDNR v0.3.5.6 style profiles (Full Model Look is default; or `Custom` for manual sliders). |
| **Performance** | `neural_placement` | `After upscaling` \| `Before upscaling` | `After upscaling` | Stage at which Neural Rendering executes (After upscaling is default for Full Model Look). |
| **Performance** | `nr_resolution` | `INT` (10 - 200%) | `100` | Working resolution percentage for the neural network. |
| **Performance** | `neural_passes` | `INT` (1 - 3) | `1` | Number of neural refinement passes (1 = fast pass, 2-3 = deeper detail). |
| **Performance** | `full_network` | `BOOLEAN` | `True` | Run the complete 71-block neural model (lmxxf runtime). |
| **Performance** | `fast_mode` | `BOOLEAN` | `False` | Enable low-latency execution path. |
| **Performance** | `dynamic_nr_resolution` | `BOOLEAN` | `False` | Dynamically scale neural resolution under load. |
| **Performance** | `model_interleave` | `Off` \| `On` | `Off` | Interleave neural evaluations across alternating frames. |
| **Quality** | `residual_strength` | `FLOAT` (0.00 - 2.00) | `1.00` | High-frequency transfer strength (`TransferStrength`). |
| **Quality** | `residual_limit` | `FLOAT` (0.00 - 1.00) | `0.25` | Maximum ratio clamp (`MaxRatio`) to prevent ringing/haloing. |
| **Quality** | `temporal_stability`| `FLOAT` (0.00 - 1.00) | `0.60` | Temporal stability smoothing to suppress inter-frame flicker. |
| **Quality** | `sharpening` | `FLOAT` (0.00 - 1.00) | `0.00` | FidelityFX CAS (Contrast Adaptive Sharpening) filter strength. |
| **Image Look** | `colour_composition`| `Classic` \| `RenoDX` | `Classic` | Color reconstruction tone operator. |
| **Image Look** | `detail_strength` | `FLOAT` (0.00 - 2.00) | `1.00` | Local micro-contrast and micro-geometry definition. |
| **Image Look** | `colour_strength` | `FLOAT` (0.00 - 2.00) | `1.00` | Chroma fidelity and saturation balance. |
| **Backend** | `backend` | `amd_nr_bridge` \| `zluda_tensor` | `amd_nr_bridge` | `amd_nr_bridge` for D3D12/OptiScaler bridge; `zluda_tensor` for pure GPU PyTorch. |
| **Backend** | `motion` | `auto` \| `optical_flow` \| `none` | `auto` | Dense motion generation via OpenCV DIS optical flow. |
| **Backend** | `keep_audio` | `on` \| `off` | `on` | Preserve original audio tracks in video output. |

---

## 🎨 Ready-to-Use Workflow

We provide a complete, pre-configured workflow located at:
📁 **[`examples/workflow_dlss5_amd.json`](examples/workflow_dlss5_amd.json)**

### How to use:
1. Open ComfyUI in your browser.
2. Drag and drop `examples/workflow_dlss5_amd.json` onto the canvas.
3. The workflow connects:
   - **`RH_DLSS5Enhance_AMD`:** Set `backend` to `amd_nr_bridge` (or `zluda_tensor`). Adjust `nr_style` (*Default*, *Cinematic*, *Crisp*, *Natural*, *Vivid*), `upscaling_mode`, and `residual_strength`.
   - **`RH_DLSS5FrameInterpolation_AMD`:** Set `output_fps` to `2x`, `3x`, or custom target FPS. Motion is automatically handled by OpenCV DIS optical flow with high fidelity.

---

## 🛠️ Troubleshooting & Pro Tips

<details>
<summary><b>🔧 1. Recommended ZLUDA startup flags</b></summary>

When running intensive Neural Rendering or Frame Interpolation pipelines under ZLUDA, enable dynamic VRAM and disable smart memory swapping for optimal latency and memory stability:
```bat
@echo off
set ZLUDA_ENABLE_DYNAMIC_VRAM=1
set HIP_VISIBLE_DEVICES=0
python main.py --disable-smart-memory --highvram
```
`--disable-smart-memory` prevents aggressive system RAM swapping across the translation layer.
</details>

<details>
<summary><b>🔧 2. Choosing between amd_nr_bridge and zluda_tensor</b></summary>

- **`auto` (Recommended):** Detects if the D3D12/HIP runtime files are present in `ComfyUI/models/dlss5/`. If present, it launches `amd_nr_bridge`. If not, it falls back transparently to `zluda_tensor`.
- **`zluda_tensor`:** Always forces pure PyTorch GPU tensor execution. Ideal for zero-dependency setups or debugging.
- **`amd_nr_bridge`:** Forces execution through the D3D12 proxy bridge (`OptiScaler` / `danielblnc` / `LmxxfNrRuntime`).
</details>

<details>
<summary><b>🔧 3. Optical Flow Performance on AMD</b></summary>

In `RH_DLSS5FrameInterpolation_AMD`, `motion="auto"` automatically selects OpenCV DIS optical flow on AMD GPUs. OpenCV DIS runs ultra-fast gradient-descent inverse composition. If you notice artifacts around fast cuts, adjust `scene_change_threshold` (default `0.24`) to trigger scene-cut protection.
</details>

---

## ☕ Support This Project / Buy Me a Coffee

Bringing cutting-edge AI technologies designed exclusively for NVIDIA hardware over to AMD Radeon graphics cards under Windows takes immense reverse-engineering, C++ bridge tuning, and persistent debugging.

If this project saved you hours of troubleshooting, gave your Radeon GPU new superpowers, or improved your ComfyUI workflow, consider buying me a coffee to support continued development and updates!

<div align="center">

<a href="https://paypal.me/mnecstream" target="_blank">
  <img src="https://img.shields.io/badge/Donate-PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white" height="42" alt="Donate with PayPal"/>
</a>

<p>
  <b>Every contribution helps keep this repository actively maintained and tested!</b><br/>
  👉 <a href="https://paypal.me/mnecstream">paypal.me/mnecstream</a>
</p>

</div>

---

## 🤝 Licenses & Shoutouts

This project is built upon the incredible work of researchers and open-source pioneers across the community:

- **[RH-RunningHub](https://github.com/RH-RunningHub/ComfyUI-RH-DLSS5):** Original creators of `ComfyUI-RH-DLSS5` (released under the MIT License).
- **[danielblnc (Daniel Blanco)](https://github.com/danielblnc/DLSS-NR-on-AMD):** Author of the groundbreaking AMD HIP native kernel implementation for DLSS-NR on Radeon GPUs.
- **[3zwr1](https://github.com/3zwr1/AMD-NR---OptiScaler) & [lmxxf](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting):** Authors of the `AMD-NR---OptiScaler` bridge and D3D12 porting layer.
- **[Speedstu](https://github.com/Speedstu/CUDA-for-AMD-Windows) & the ZLUDA project:** For making CUDA/HIP acceleration viable on AMD GPUs in Windows.

---

<div align="center">
  <b>ComfyUI-RH-DLSS5-AMD</b> • Crafted with passion for the open AMD AI community.
</div>
