#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include "nvsdk_ngx_defs.h"
#include "nvsdk_ngx_params.h"

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

// --- NGX Function Pointer Types ---
typedef NVSDK_NGX_Result(__cdecl* PFN_NVSDK_NGX_D3D12_Init_with_ProjectID)(const char*, NVSDK_NGX_EngineType, const char*, const wchar_t*, ID3D12Device*, const NVSDK_NGX_FeatureCommonInfo*, NVSDK_NGX_Version);
typedef NVSDK_NGX_Result(__cdecl* PFN_NVSDK_NGX_D3D12_Shutdown)(void);
typedef NVSDK_NGX_Result(__cdecl* PFN_NVSDK_NGX_D3D12_AllocateParameters)(NVSDK_NGX_Parameter**);
typedef NVSDK_NGX_Result(__cdecl* PFN_NVSDK_NGX_D3D12_GetCapabilityParameters)(NVSDK_NGX_Parameter**);
typedef NVSDK_NGX_Result(__cdecl* PFN_NVSDK_NGX_D3D12_CreateFeature)(ID3D12GraphicsCommandList*, NVSDK_NGX_Feature, NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
typedef NVSDK_NGX_Result(__cdecl* PFN_NVSDK_NGX_D3D12_ReleaseFeature)(NVSDK_NGX_Handle*);
typedef NVSDK_NGX_Result(__cdecl* PFN_NVSDK_NGX_D3D12_EvaluateFeature)(ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*, const NVSDK_NGX_Parameter*, void*);

// --- Globals ---
HMODULE g_PassModules[3] = {};
volatile bool g_WatchdogRunning = true;
int g_TotalFramesProcessed = 0; // Tracks total frames across all FRAME commands

ID3D12Device* g_pDevice = nullptr;
ID3D12CommandQueue* g_pCommandQueue = nullptr;
ID3D12CommandAllocator* g_pCommandAllocator = nullptr;
ID3D12GraphicsCommandList* g_pCommandList = nullptr;
ID3D12Fence* g_pFence = nullptr;
UINT64 g_FenceValue = 0;
HANDLE g_FenceEvent = nullptr;

// Use stderr for all debug/diagnostic output; stdout ONLY for protocol messages
#define LOG(x) do { std::cerr << x << std::endl; } while(0)

// =====================================================================
// Watchdog Patches (from EnvyDynamicPacing.asi / lib.rs)
// =====================================================================
bool PatchMemory(HMODULE hMod, DWORD rva, const std::vector<uint8_t>& expected, const std::vector<uint8_t>& patch) {
    if (!hMod) return false;
    uint8_t* target = (uint8_t*)hMod + rva;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (target[i] != expected[i]) {
            LOG("  Patch at RVA 0x" << std::hex << rva << " SKIPPED (byte mismatch at offset " << i << ": expected 0x" << (int)expected[i] << ", got 0x" << (int)target[i] << ")");
            return false;
        }
    }
    DWORD oldProtect;
    if (VirtualProtect(target, patch.size(), PAGE_EXECUTE_READWRITE, &oldProtect)) {
        memcpy(target, patch.data(), patch.size());
        VirtualProtect(target, patch.size(), oldProtect, &oldProtect);
        return true;
    }
    return false;
}

int PatchPassModule(HMODULE hMod, const char* name) {
    if (!hMod) return 0;
    LOG("Patching " << name << "...");
    int count = 0;

    // 1. Init keep_input_flag at RVA 0x1061E: -1 -> 0
    uint8_t* p = (uint8_t*)hMod + 0x1061E;
    if (p[0] == 0xC7 && p[1] == 0x05) {
        DWORD old; VirtualProtect(p + 6, 4, PAGE_EXECUTE_READWRITE, &old);
        *(uint32_t*)(p + 6) = 0;
        VirtualProtect(p + 6, 4, old, &old);
        count++;
    }
    // 2. NOP timeout branch
    if (PatchMemory(hMod, 0xF0D1, {0x75, 0x6A}, {0x90, 0x90})) count++;
    // 3. Timeout write MOV EAX,-1 -> MOV EAX,0
    p = (uint8_t*)hMod + 0xF293;
    if (p[0] == 0xB8) {
        DWORD old; VirtualProtect(p + 1, 4, PAGE_EXECUTE_READWRITE, &old);
        *(uint32_t*)(p + 1) = 0;
        VirtualProtect(p + 1, 4, old, &old);
        count++;
    }
    // 4-8. NOP watchdog/budget/cap jumps
    if (PatchMemory(hMod, 0xF0DF, {0x0F, 0x8D}, {0x90, 0x90, 0x90, 0x90, 0x90, 0x90})) count++;
    if (PatchMemory(hMod, 0xF0F7, {0x0F, 0x8C}, {0x90, 0x90, 0x90, 0x90, 0x90, 0x90})) count++;
    if (PatchMemory(hMod, 0xF0BF, {0x87, 0x05}, {0x90, 0x90, 0x90, 0x90, 0x90, 0x90})) count++;
    if (PatchMemory(hMod, 0xF1B7, {0x87, 0x15}, {0x90, 0x90, 0x90, 0x90, 0x90, 0x90})) count++;
    if (PatchMemory(hMod, 0xF121, {0x87, 0x05}, {0x90, 0x90, 0x90, 0x90, 0x90, 0x90})) count++;
    // 9. Init cap -> 2 billion
    p = (uint8_t*)hMod + 0x10568;
    if (p[0] == 0xC7 && p[1] == 0x05) {
        DWORD old; VirtualProtect(p + 6, 4, PAGE_EXECUTE_READWRITE, &old);
        *(uint32_t*)(p + 6) = 2000000000;
        VirtualProtect(p + 6, 4, old, &old);
        count++;
    }

    // Initial RAM clamp
    uint8_t* base = (uint8_t*)hMod;
    DWORD oldP;
    if (VirtualProtect(base + 0x76C44, 0x120, PAGE_EXECUTE_READWRITE, &oldP)) {
        *(uint32_t*)(base + 0x76C44) = 2000000000;
        *(uint32_t*)(base + 0x76C54) = 2000;
        *(int32_t*)(base + 0x76D54) = 0;
        VirtualProtect(base + 0x76C44, 0x120, oldP, &oldP);
    }

    LOG("  " << name << ": " << count << "/9 code patches applied, RAM clamped.");
    return count;
}

int PatchOptiScaler(HMODULE hMod, const char* name) {
    if (!hMod) return 0;
    LOG("Patching OptiScaler (" << name << ")...");
    int count = 0;
    // Disable DlssNr ExposureScan Adopt (RVA 0x3F2980): prevents scanning & adopting random buffers as exposure
    if (PatchMemory(hMod, 0x3F2980, {0x40, 0x55, 0x53, 0x56}, {0xC3, 0x90, 0x90, 0x90})) count++;
    // Disable DlssNr ExposureScan Tick (RVA 0x3F2BA0): prevents tracking candidate resources
    if (PatchMemory(hMod, 0x3F2BA0, {0x48, 0x89, 0x5C, 0x24}, {0xC3, 0x90, 0x90, 0x90})) count++;
    LOG("  OptiScaler: " << count << "/2 exposure-scan patches applied.");
    return count;
}



// =====================================================================
// D3D12 Helpers
// =====================================================================
void WaitForGPU() {
    g_FenceValue++;
    g_pCommandQueue->Signal(g_pFence, g_FenceValue);
    if (g_pFence->GetCompletedValue() < g_FenceValue) {
        g_pFence->SetEventOnCompletion(g_FenceValue, g_FenceEvent);
        WaitForSingleObject(g_FenceEvent, INFINITE);
    }
}

void TransitionResource(ID3D12GraphicsCommandList* cmd, ID3D12Resource* res, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    if (!res) return;
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = res;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd->ResourceBarrier(1, &barrier);
}

ID3D12Resource* CreateTexture(UINT width, UINT height, DXGI_FORMAT format, D3D12_RESOURCE_STATES initialState, bool uav = false) {
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width; desc.Height = height;
    desc.DepthOrArraySize = 1; desc.MipLevels = 1;
    desc.Format = format; desc.SampleDesc.Count = 1;
    desc.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    ID3D12Resource* r = nullptr;
    HRESULT hr = g_pDevice->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, initialState, nullptr, IID_PPV_ARGS(&r));
    if (FAILED(hr)) LOG("CreateTexture failed: 0x" << std::hex << hr);
    return r;
}

ID3D12Resource* CreateBuffer(UINT64 size, D3D12_HEAP_TYPE heapType) {
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size; desc.Height = 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN; desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = heapType;
    ID3D12Resource* r = nullptr;
    g_pDevice->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc,
        heapType == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr, IID_PPV_ARGS(&r));
    return r;
}

void UploadTexture(ID3D12Resource* pDest, UINT width, UINT height, DXGI_FORMAT format, UINT bpp, const void* pData) {
    UINT rowPitch = width * bpp;
    UINT uploadPitch = (rowPitch + 255) & ~255;
    ID3D12Resource* pUpload = CreateBuffer((UINT64)uploadPitch * height, D3D12_HEAP_TYPE_UPLOAD);
    void* pMapped = nullptr;
    pUpload->Map(0, nullptr, &pMapped);
    for (UINT y = 0; y < height; ++y)
        memcpy((uint8_t*)pMapped + y * uploadPitch, (uint8_t*)pData + y * rowPitch, rowPitch);
    pUpload->Unmap(0, nullptr);

    g_pCommandAllocator->Reset();
    g_pCommandList->Reset(g_pCommandAllocator, nullptr);
    D3D12_TEXTURE_COPY_LOCATION destLoc = {}; destLoc.pResource = pDest; destLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION srcLoc = {}; srcLoc.pResource = pUpload; srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLoc.PlacedFootprint.Footprint = { format, width, height, 1, uploadPitch };
    TransitionResource(g_pCommandList, pDest, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    g_pCommandList->CopyTextureRegion(&destLoc, 0, 0, 0, &srcLoc, nullptr);
    TransitionResource(g_pCommandList, pDest, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    g_pCommandList->Close();
    ID3D12CommandList* ppLists[] = { g_pCommandList };
    g_pCommandQueue->ExecuteCommandLists(1, ppLists);
    WaitForGPU();
    pUpload->Release();
}

void DownloadTexture(ID3D12Resource* pSrc, UINT width, UINT height, DXGI_FORMAT format, UINT bpp, void* pData) {
    g_pCommandAllocator->Reset();
    g_pCommandList->Reset(g_pCommandAllocator, nullptr);
    UINT rowPitch = width * bpp;
    UINT readbackPitch = (rowPitch + 255) & ~255;
    ID3D12Resource* pReadback = CreateBuffer((UINT64)readbackPitch * height, D3D12_HEAP_TYPE_READBACK);
    if (!pReadback) { LOG("DownloadTexture: CreateBuffer failed!"); return; }
    TransitionResource(g_pCommandList, pSrc, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION srcLoc = {}; srcLoc.pResource = pSrc; srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION destLoc = {}; destLoc.pResource = pReadback; destLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destLoc.PlacedFootprint.Footprint = { format, width, height, 1, readbackPitch };
    g_pCommandList->CopyTextureRegion(&destLoc, 0, 0, 0, &srcLoc, nullptr);
    g_pCommandList->Close();
    ID3D12CommandList* ppLists[] = { g_pCommandList };
    g_pCommandQueue->ExecuteCommandLists(1, ppLists);
    WaitForGPU();
    void* pMapped = nullptr;
    if (SUCCEEDED(pReadback->Map(0, nullptr, &pMapped))) {
        for (UINT y = 0; y < height; ++y)
            memcpy((uint8_t*)pData + y * rowPitch, (uint8_t*)pMapped + y * readbackPitch, rowPitch);
        pReadback->Unmap(0, nullptr);
    }
    pReadback->Release();
}

std::vector<uint8_t> ReadFileBin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return {};
    size_t size = f.tellg(); f.seekg(0);
    std::vector<uint8_t> data(size);
    f.read((char*)data.data(), size);
    return data;
}

// Camera matrices for Ray Reconstruction (Feature 13 DLSS-D / FSR-RR)
static float g_WorldToView[16] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
};

static float g_ViewToClip[16] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
};

void InitCameraMatrices(UINT w, UINT h) {
    float fov = 1.04719755f; // 60 deg
    float aspect = (float)w / (float)h;
    float nearZ = 0.1f;
    float farZ = 100.0f;
    float tanHalfFov = tanf(fov / 2.0f);

    memset(g_ViewToClip, 0, sizeof(g_ViewToClip));
    g_ViewToClip[0] = 1.0f / (aspect * tanHalfFov);
    g_ViewToClip[5] = 1.0f / tanHalfFov;
    // DirectX 12 Reversed-Z projection matrix coefficients:
    // Ensures OptiScaler extracts correct near=0.1, far=100.0 depth bounds so backgrounds are razor sharp
    g_ViewToClip[10] = -nearZ / (farZ - nearZ);
    g_ViewToClip[11] = 1.0f;
    g_ViewToClip[14] = (nearZ * farZ) / (farZ - nearZ);
    g_ViewToClip[15] = 0.0f;
}

// =====================================================================
// Run one evaluation pass (one "frame" of the simulated game loop)
// =====================================================================
void RunOneEvalPass(
    NVSDK_NGX_Parameter* pParams, const NVSDK_NGX_Handle* pHandle,
    PFN_NVSDK_NGX_D3D12_EvaluateFeature pfnEval,
    ID3D12Resource* pOutput, ID3D12Resource* pColor,
    ID3D12Resource* pDepth, ID3D12Resource* pMvec,
    ID3D12Resource* pNormal, ID3D12Resource* pAlbedo,
    ID3D12Resource* pExposure, UINT w, UINT h, UINT out_w, UINT out_h,
    IDXGISwapChain3* pSwapChain, int frameIndex, bool resetHistory)
{
    g_pCommandAllocator->Reset();
    g_pCommandList->Reset(g_pCommandAllocator, nullptr);

    pParams->Set(NVSDK_NGX_Parameter_Reset, resetHistory ? 1 : 0);
    pParams->Set("DLSSNR.Reset", resetHistory ? 1 : 0);
    pParams->Set("DLSSG.Reset", resetHistory ? 1 : 0);

    // Rebind all textures and buffers on every pass (ensures complete DX12 state)
    pParams->Set(NVSDK_NGX_Parameter_Color, pColor);
    pParams->Set("DLSSNR.Color", pColor);
    pParams->Set("DLSSD.Color", pColor);
    pParams->Set("DLSSNR.ColorSubrectBaseX", 0);
    pParams->Set("DLSSNR.ColorSubrectBaseY", 0);
    pParams->Set("DLSSNR.ColorSubrectWidth", w);
    pParams->Set("DLSSNR.ColorSubrectHeight", h);

    pParams->Set(NVSDK_NGX_Parameter_Output, pOutput);
    pParams->Set("DLSSNR.Output", pOutput);
    pParams->Set("DLSSNR.Backbuffer", pOutput);
    pParams->Set("DLSSD.Output", pOutput);
    pParams->Set("DLSSNR.OutputSubrectBaseX", 0);
    pParams->Set("DLSSNR.OutputSubrectBaseY", 0);
    pParams->Set("DLSSNR.OutputSubrectWidth", out_w);
    pParams->Set("DLSSNR.OutputSubrectHeight", out_h);

    pParams->Set(NVSDK_NGX_Parameter_Depth, pDepth);
    pParams->Set("DLSSNR.Depth", pDepth);
    pParams->Set("DLSSD.Depth", pDepth);
    pParams->Set("DepthInverted", 1);
    pParams->Set("DLSSNR.DepthInverted", 1);
    pParams->Set("DLSSG.DepthInverted", 1);
    pParams->Set("DLSS.Input.Depth.Subrect.Base.X", 0);
    pParams->Set("DLSS.Input.Depth.Subrect.Base.Y", 0);
    pParams->Set("DLSSNR.DepthSubrectBaseX", 0);
    pParams->Set("DLSSNR.DepthSubrectBaseY", 0);
    pParams->Set("DLSSNR.DepthSubrectWidth", w);
    pParams->Set("DLSSNR.DepthSubrectHeight", h);

    pParams->Set(NVSDK_NGX_Parameter_MotionVectors, pMvec);
    pParams->Set("DLSSNR.MVec", pMvec);
    pParams->Set("DLSSD.MotionVectors", pMvec);
    pParams->Set("DLSS.Input.MV.Subrect.Base.X", 0);
    pParams->Set("DLSS.Input.MV.Subrect.Base.Y", 0);
    pParams->Set("DLSSNR.MVecSubrectBaseX", 0);
    pParams->Set("DLSSNR.MVecSubrectBaseY", 0);
    pParams->Set("DLSSNR.MVecSubrectWidth", out_w);
    pParams->Set("DLSSNR.MVecSubrectHeight", out_h);
    pParams->Set(NVSDK_NGX_Parameter_MV_Scale_X, 1.0f);
    pParams->Set(NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);
    pParams->Set("DLSSNR.MVecScaleX", 1.0f);
    pParams->Set("DLSSNR.MVecScaleY", 1.0f);

    pParams->Set("Roughness", pNormal);
    pParams->Set("GBuffer.Normals", pNormal);
    pParams->Set("GBuffer.Roughness", pNormal);
    pParams->Set("DLSS.Input.Normals", pNormal);
    pParams->Set("DLSS.Input.Roughness", pNormal);
    pParams->Set("InNormals", pNormal);
    pParams->Set("OutNormals", pNormal);
    pParams->Set("FfxDenoiserNormalsInViewSpace", 1);

    pParams->Set("Albedo", pAlbedo);
    pParams->Set("GBuffer.Albedo", pAlbedo);
    pParams->Set("GBuffer.DiffuseAlbedo", pAlbedo);
    pParams->Set("GBuffer.SpecularAlbedo", pAlbedo);
    pParams->Set("DLSS.Input.DiffuseAlbedo", pAlbedo);
    pParams->Set("DLSS.Input.SpecularAlbedo", pAlbedo);
    pParams->Set("InDiffAlbedo", pAlbedo);
    pParams->Set("InSpecAlbedo", pAlbedo);
    pParams->Set("FfxDenoiserAlbedoFp16", 1);
    pParams->Set("DLSS.Input.Bias.Current.Color.Mask", pAlbedo);

    pParams->Set(NVSDK_NGX_Parameter_ExposureTexture, pExposure);
    pParams->Set("PreExposure", 1.0f);
    pParams->Set(NVSDK_NGX_Parameter_Sharpness, 0.0f);

    // Camera Matrices for FSR-RR (Ray Reconstruction Feature 13)
    pParams->Set("WorldToViewMatrix", (void*)g_WorldToView);
    pParams->Set("ViewToClipMatrix", (void*)g_ViewToClip);
    pParams->Set("DLSS.Camera.WorldToViewMatrix", (void*)g_WorldToView);
    pParams->Set("DLSS.Camera.ViewToClipMatrix", (void*)g_ViewToClip);
    pParams->Set("FSR.cameraNear", 0.1f);
    pParams->Set("FSR.cameraFar", 1000.0f);
    pParams->Set("FSR.cameraFovAngleVertical", 1.04719755f);
    pParams->Set("CameraNear", 0.1f);
    pParams->Set("CameraFar", 1000.0f);
    pParams->Set("CameraFOV", 60.0f);

    TransitionResource(g_pCommandList, pOutput, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // Halton(2,3) jitter
    float f2 = 1.0f, r2 = 0.0f; int idx2 = frameIndex + 1;
    while (idx2 > 0) { f2 /= 2.0f; r2 += f2 * (idx2 % 2); idx2 /= 2; }
    float f3 = 1.0f, r3 = 0.0f; int idx3 = frameIndex + 1;
    while (idx3 > 0) { f3 /= 3.0f; r3 += f3 * (idx3 % 3); idx3 /= 3; }
    pParams->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, r2 - 0.5f);
    pParams->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, r3 - 0.5f);
    pParams->Set("Jitter.Offset.X", r2 - 0.5f);
    pParams->Set("Jitter.Offset.Y", r3 - 0.5f);
    pParams->Set("DLSSG.JitterOffsetX", r2 - 0.5f);
    pParams->Set("DLSSG.JitterOffsetY", r3 - 0.5f);

    NVSDK_NGX_Result res = pfnEval(g_pCommandList, pHandle, pParams, nullptr);
    if (NVSDK_NGX_FAILED(res)) {
        LOG("EvaluateFeature failed: 0x" << std::hex << res);
    }

    TransitionResource(g_pCommandList, pOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    g_pCommandList->Close();
    g_pCommandQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&g_pCommandList);

    MSG msg;
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessage(&msg); }
    if (pSwapChain) pSwapChain->Present(0, 0);

    WaitForGPU();
}

// Detect HIP device 0 LUID to bind D3D12 device to the identical physical GPU adapter
bool GetHipDeviceLuid(LUID* outLuid) {
    if (!outLuid) return false;
    HMODULE hHip = LoadLibraryA("amdhip64_7.dll");
    if (!hHip) hHip = LoadLibraryA("amdhip64.dll");
    if (!hHip) return false;

    typedef int (*pfn_hipInit)(unsigned int);
    typedef int (*pfn_hipGetDeviceCount)(int*);
    typedef int (*pfn_hipGetDevicePropertiesR0600)(void*, int);

    auto pfnInit = (pfn_hipInit)GetProcAddress(hHip, "hipInit");
    auto pfnCount = (pfn_hipGetDeviceCount)GetProcAddress(hHip, "hipGetDeviceCount");
    auto pfnProps = (pfn_hipGetDevicePropertiesR0600)GetProcAddress(hHip, "hipGetDevicePropertiesR0600");

    if (!pfnInit || !pfnCount || !pfnProps) {
        FreeLibrary(hHip);
        return false;
    }

    pfnInit(0);
    int devCount = 0;
    if (pfnCount(&devCount) != 0 || devCount <= 0) {
        FreeLibrary(hHip);
        return false;
    }

    std::vector<uint8_t> propBuf(8192, 0);
    if (pfnProps(propBuf.data(), 0) == 0) {
        // LUID is located at offset 272 (0x110) in hipDeviceProp_t (R0600)
        LUID* pLuid = (LUID*)(propBuf.data() + 272);
        if (pLuid->LowPart != 0 || pLuid->HighPart != 0) {
            *outLuid = *pLuid;
            FreeLibrary(hHip);
            return true;
        }
    }

    FreeLibrary(hHip);
    return false;
}

// =====================================================================
// MAIN
// =====================================================================
int main(int argc, char** argv) {
    // Usage: dlss_bridge <w> <h> <out_w> <out_h> <color.bin> <depth.bin> <mvec.bin> <output.bin> <passes> <normal.bin> <albedo.bin> <fps> <frametime> <warmup>
    // In daemon mode, after init + warmup, reads FRAME commands from stdin.
    if (argc < 15) {
        LOG("Usage: dlss_bridge w h out_w out_h color.bin depth.bin mvec.bin output.bin passes normal.bin albedo.bin fps frametime warmup");
        return 1;
    }

    Sleep(200);

    UINT w = std::stoi(argv[1]), h = std::stoi(argv[2]);
    UINT out_w = std::stoi(argv[3]), out_h = std::stoi(argv[4]);
    int passes = std::stoi(argv[9]);
    int target_fps = std::stoi(argv[12]);
    float process_time_sec = std::stof(argv[13]);
    int warmup_passes = std::stoi(argv[14]);
    int settle_pause_ms = (argc > 15) ? std::stoi(argv[15]) : 350;
    int warmup_interval_ms = (argc > 16) ? std::stoi(argv[16]) : 50;
    int feature_id = (argc > 17) ? std::stoi(argv[17]) : 1;
    const char* envFeature = getenv("DLSS_FEATURE_ID");
    if (envFeature && envFeature[0] != '\0') {
        try { feature_id = std::stoi(envFeature); } catch (...) {}
    }

    std::string colorPath = argv[5], depthPath = argv[6], mvecPath = argv[7];
    std::string outputPath = argv[8], normalPath = argv[10], albedoPath = argv[11];

    // OptiScaler (AmdPreSr) manages, loads, patches, and isolates dlssnr_amd_pass*.dll / LmxxfNrRuntime.dll dynamically.


    // --- Read initial frame data ---
    std::vector<uint8_t> colorData = ReadFileBin(colorPath);
    std::vector<uint8_t> depthData = ReadFileBin(depthPath);
    std::vector<uint8_t> mvecData  = ReadFileBin(mvecPath);
    if (colorData.empty()) { LOG("Bad color data"); return 1; }

    // --- D3D12 Init ---
    LOG("Initializing D3D12...");
    IDXGIFactory4* pFactory = nullptr;
    CreateDXGIFactory1(IID_PPV_ARGS(&pFactory));

    LUID hipLuid = {};
    bool hasHipLuid = GetHipDeviceLuid(&hipLuid);
    if (hasHipLuid) {
        LOG("Detected active HIP device LUID: 0x" << std::hex << hipLuid.HighPart << ":" << hipLuid.LowPart << std::dec);
    }

    IDXGIAdapter1* pAdapter = nullptr;
    IDXGIAdapter1* pFallbackAdapter = nullptr;

    for (UINT i = 0; pFactory->EnumAdapters1(i, &pAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc; pAdapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { pAdapter->Release(); pAdapter = nullptr; continue; }

        if (hasHipLuid && desc.AdapterLuid.LowPart == hipLuid.LowPart && desc.AdapterLuid.HighPart == hipLuid.HighPart) {
            LOG("Selected matching DXGI Adapter " << i << " for HIP device: " << desc.Description
                << " (LUID 0x" << std::hex << desc.AdapterLuid.HighPart << ":" << desc.AdapterLuid.LowPart << std::dec << ")");
            if (pFallbackAdapter) { pFallbackAdapter->Release(); pFallbackAdapter = nullptr; }
            break;
        }

        if (!pFallbackAdapter) {
            pFallbackAdapter = pAdapter;
        } else {
            pAdapter->Release();
        }
        pAdapter = nullptr;
    }

    if (!pAdapter) {
        pAdapter = pFallbackAdapter;
    }
    if (!pAdapter) { LOG("No hardware adapter found"); return 1; }

    HRESULT hr = D3D12CreateDevice(pAdapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&g_pDevice));
    if (FAILED(hr)) { LOG("D3D12 Device creation failed"); return 1; }

    D3D12_COMMAND_QUEUE_DESC qd = {}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    g_pDevice->CreateCommandQueue(&qd, IID_PPV_ARGS(&g_pCommandQueue));
    g_pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g_pCommandAllocator));
    g_pDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_pCommandAllocator, nullptr, IID_PPV_ARGS(&g_pCommandList));
    g_pCommandList->Close();
    g_pDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_pFence));
    g_FenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    // --- Monitor Enumeration for Kanvas 20 (Secondary Monitor) ---
    struct MonitorTarget {
        int x = 0;
        int y = 0;
        int width = 1536;
        int height = 864;
        bool foundSecondary = false;
        char deviceName[32] = {};
    };

    auto EnumMonProc = [](HMONITOR hMon, HDC, LPRECT, LPARAM lp) -> BOOL {
        auto* target = reinterpret_cast<MonitorTarget*>(lp);
        MONITORINFOEXA mi = {};
        mi.cbSize = sizeof(mi);
        if (GetMonitorInfoA(hMon, &mi)) {
            if (!(mi.dwFlags & MONITORINFOF_PRIMARY)) {
                target->x = mi.rcMonitor.left;
                target->y = mi.rcMonitor.top;
                target->width = mi.rcMonitor.right - mi.rcMonitor.left;
                target->height = mi.rcMonitor.bottom - mi.rcMonitor.top;
                target->foundSecondary = true;
                strncpy_s(target->deviceName, mi.szDevice, sizeof(target->deviceName) - 1);
                return FALSE; // Found secondary display (Kanvas 20)
            } else if (!target->foundSecondary) {
                target->x = mi.rcMonitor.left;
                target->y = mi.rcMonitor.top;
                target->width = mi.rcMonitor.right - mi.rcMonitor.left;
                target->height = mi.rcMonitor.bottom - mi.rcMonitor.top;
                strncpy_s(target->deviceName, mi.szDevice, sizeof(target->deviceName) - 1);
            }
        }
        return TRUE;
    };

    MonitorTarget monTarget;
    EnumDisplayMonitors(nullptr, nullptr, EnumMonProc, reinterpret_cast<LPARAM>(&monTarget));
    LOG("Target display for bridge window: " << (monTarget.foundSecondary ? "Secondary Monitor (Kanvas 20)" : "Primary Display")
        << " [" << monTarget.deviceName << "] at (" << monTarget.x << ", " << monTarget.y
        << ") size " << monTarget.width << "x" << monTarget.height);

    int winW = (int)out_w;
    int winH = (int)out_h;
    if (monTarget.width > 0 && winW > monTarget.width) winW = monTarget.width;
    if (monTarget.height > 0 && winH > monTarget.height) winH = monTarget.height;
    int winX = monTarget.x;
    int winY = monTarget.y;

    WNDCLASSA wc = {};
    wc.lpfnWndProc = DefWindowProcA;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = "ED2Bridge";
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    RegisterClassA(&wc);

    // Create a REAL visible foreground window on Kanvas 20 (secondary screen)
    // so OptiScaler / AMD-NR has an active foreground swapchain and does not stall/freeze.
    HWND hWnd = CreateWindowExA(
        WS_EX_APPWINDOW,
        "ED2Bridge",
        "DLSS5-AMD Neural Rendering Bridge (Kanvas 20)",
        WS_POPUP | WS_VISIBLE,
        winX, winY, winW, winH,
        nullptr, nullptr, wc.hInstance, nullptr
    );
    ShowWindow(hWnd, SW_SHOW);
    UpdateWindow(hWnd);
    SetForegroundWindow(hWnd);

    DXGI_SWAP_CHAIN_DESC1 scd = {};
    scd.Width = winW; scd.Height = winH; scd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.SampleDesc.Count = 1; scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.BufferCount = 2; scd.Scaling = DXGI_SCALING_STRETCH;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

    IDXGISwapChain1* pSC1 = nullptr;
    pFactory->CreateSwapChainForHwnd(g_pCommandQueue, hWnd, &scd, nullptr, nullptr, &pSC1);
    IDXGISwapChain3* pSwapChain = nullptr;
    if (pSC1) pSC1->QueryInterface(IID_PPV_ARGS(&pSwapChain));

    // --- Create Textures (persistent — reused across all frames) ---
    LOG("Creating GPU textures...");
    ID3D12Resource* pColor = CreateTexture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_COPY_DEST, true);
    ID3D12Resource* pDepth = CreateTexture(w, h, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* pMvec  = CreateTexture(out_w, out_h, DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* pAlbedo = CreateTexture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* pNormal = CreateTexture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* pExposure = CreateTexture(1, 1, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* pOutput = CreateTexture(out_w, out_h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_COMMON, true);

    // Upload initial frame data
    UploadTexture(pColor, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, colorData.data());
    UploadTexture(pDepth, w, h, DXGI_FORMAT_R32_FLOAT, 4, depthData.data());
    UploadTexture(pMvec, out_w, out_h, DXGI_FORMAT_R16G16_FLOAT, 4, mvecData.data());

    std::vector<uint8_t> albedoData = ReadFileBin(albedoPath);
    std::vector<uint8_t> normalData = ReadFileBin(normalPath);
    if (albedoData.empty()) { albedoData.resize(w * h * 8, 0); for (size_t i = 0; i < w*h; i++) { ((uint16_t*)albedoData.data())[i*4+3] = 0x3C00; } }
    if (normalData.empty()) { normalData.resize(w * h * 8, 0); for (size_t i = 0; i < w*h; i++) { ((uint16_t*)normalData.data())[i*4+2] = 0x3C00; } }
    UploadTexture(pAlbedo, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, albedoData.data());
    UploadTexture(pNormal, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, normalData.data());
    float expVal = 1.0f;
    UploadTexture(pExposure, 1, 1, DXGI_FORMAT_R32_FLOAT, 4, &expVal);

    // --- Init NGX ---
    LOG("Loading OptiScaler...");
    HMODULE hNvngx = GetModuleHandleA("dxgi.dll");
    if (!hNvngx || !GetProcAddress(hNvngx, "NVSDK_NGX_D3D12_Init_with_ProjectID")) {
        hNvngx = GetModuleHandleA("OptiScaler.dll");
    }
    if (!hNvngx || !GetProcAddress(hNvngx, "NVSDK_NGX_D3D12_Init_with_ProjectID")) {
        hNvngx = LoadLibraryA("OptiScaler.dll");
    }
    if (!hNvngx || !GetProcAddress(hNvngx, "NVSDK_NGX_D3D12_Init_with_ProjectID")) {
        hNvngx = LoadLibraryA("dxgi.dll");
    }
    if (!hNvngx) { LOG("Cannot load OptiScaler.dll or dxgi.dll"); return 1; }
    PatchOptiScaler(hNvngx, "OptiScaler.dll");

    auto pfnInit = (PFN_NVSDK_NGX_D3D12_Init_with_ProjectID)GetProcAddress(hNvngx, "NVSDK_NGX_D3D12_Init_with_ProjectID");
    auto pfnAllocParams = (PFN_NVSDK_NGX_D3D12_AllocateParameters)GetProcAddress(hNvngx, "NVSDK_NGX_D3D12_AllocateParameters");
    auto pfnCreateFeature = (PFN_NVSDK_NGX_D3D12_CreateFeature)GetProcAddress(hNvngx, "NVSDK_NGX_D3D12_CreateFeature");
    auto pfnEval = (PFN_NVSDK_NGX_D3D12_EvaluateFeature)GetProcAddress(hNvngx, "NVSDK_NGX_D3D12_EvaluateFeature");
    if (!pfnInit || !pfnAllocParams || !pfnCreateFeature || !pfnEval) { LOG("Missing NGX functions"); return 1; }

    wchar_t cwdBuf[MAX_PATH]; GetCurrentDirectoryW(MAX_PATH, cwdBuf);
    NVSDK_NGX_Result res = pfnInit("DLSS5-AMD", NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", cwdBuf, g_pDevice, nullptr, (NVSDK_NGX_Version)NVSDK_NGX_VERSION_API_MACRO);
    if (NVSDK_NGX_FAILED(res)) { LOG("NGX Init failed: 0x" << std::hex << res); return 1; }

    NVSDK_NGX_Parameter* pParams = nullptr;
    pfnAllocParams(&pParams);

    pParams->Set(NVSDK_NGX_Parameter_Width, w);
    pParams->Set(NVSDK_NGX_Parameter_Height, h);
    pParams->Set(NVSDK_NGX_Parameter_OutWidth, out_w);
    pParams->Set(NVSDK_NGX_Parameter_OutHeight, out_h);
    pParams->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, w);
    pParams->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, h);
    pParams->Set(NVSDK_NGX_Parameter_PerfQualityValue, NVSDK_NGX_PerfQuality_Value_MaxQuality);
    pParams->Set(NVSDK_NGX_Parameter_Sharpness, 0.0f);
    pParams->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, (1 << 3)); // kDepthInverted
    pParams->Set("DepthInverted", 1);
    pParams->Set("DLSS.Use.HW.Depth", 1);
    pParams->Set("FfxDenoiserHardwareDepth", 1);
    pParams->Set("FfxDenoiserNormalsInViewSpace", 1);
    pParams->Set("FfxDenoiserAlbedoFp16", 1);
    pParams->Set("DLSS.Roughness.Mode", 0);
    pParams->Set("DLSS.Denoise.Mode", 0);

    // Initial binding of textures to params prior to CreateFeature
    pParams->Set(NVSDK_NGX_Parameter_Color, pColor);
    pParams->Set("DLSSNR.Color", pColor);
    pParams->Set("DLSSD.Color", pColor);
    pParams->Set(NVSDK_NGX_Parameter_Output, pOutput);
    pParams->Set("DLSSNR.Output", pOutput);
    pParams->Set("DLSSD.Output", pOutput);
    pParams->Set(NVSDK_NGX_Parameter_Depth, pDepth);
    pParams->Set("DLSSNR.Depth", pDepth);
    pParams->Set("DLSSD.Depth", pDepth);
    pParams->Set(NVSDK_NGX_Parameter_MotionVectors, pMvec);
    pParams->Set("DLSSNR.MVec", pMvec);
    pParams->Set("DLSSD.MotionVectors", pMvec);
    pParams->Set("Roughness", pNormal);
    pParams->Set("GBuffer.Normals", pNormal);
    pParams->Set("GBuffer.Roughness", pNormal);
    pParams->Set("Albedo", pAlbedo);
    pParams->Set("GBuffer.Albedo", pAlbedo);
    pParams->Set(NVSDK_NGX_Parameter_ExposureTexture, pExposure);

    // Initialize & bind Camera Matrices for FSR-RR (Feature 13)
    InitCameraMatrices(w, h);
    pParams->Set("WorldToViewMatrix", (void*)g_WorldToView);
    pParams->Set("ViewToClipMatrix", (void*)g_ViewToClip);
    pParams->Set("DLSS.Camera.WorldToViewMatrix", (void*)g_WorldToView);
    pParams->Set("DLSS.Camera.ViewToClipMatrix", (void*)g_ViewToClip);
    pParams->Set("FSR.cameraNear", 0.1f);
    pParams->Set("FSR.cameraFar", 1000.0f);
    pParams->Set("FSR.cameraFovAngleVertical", 1.04719755f);
    pParams->Set("CameraNear", 0.1f);
    pParams->Set("CameraFar", 1000.0f);
    pParams->Set("CameraFOV", 60.0f);

    // CreateFeature: select between Feature 1 (SuperSampling) and Feature 13 (Ray Reconstruction / DLSS-D / FSR-RR)
    NVSDK_NGX_Feature targetFeature = (feature_id == 13) ? NVSDK_NGX_Feature_RayReconstruction : NVSDK_NGX_Feature_SuperSampling;
    LOG("Creating NGX Feature: " << (int)targetFeature 
        << (targetFeature == NVSDK_NGX_Feature_RayReconstruction ? " (Ray Reconstruction / DLSS-D / FSR-RR Feature 13)" : " (SuperSampling / DLSS-SR / FSR-SR Feature 1)"));

    g_pCommandAllocator->Reset(); g_pCommandList->Reset(g_pCommandAllocator, nullptr);
    ID3D12CommandAllocator* pCA2 = nullptr; ID3D12GraphicsCommandList* pCL2 = nullptr;
    g_pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&pCA2));
    g_pDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, pCA2, nullptr, IID_PPV_ARGS(&pCL2));
    NVSDK_NGX_Handle* pHandle = nullptr;
    res = pfnCreateFeature(pCL2, targetFeature, pParams, &pHandle);
    if (NVSDK_NGX_FAILED(res) && targetFeature == NVSDK_NGX_Feature_RayReconstruction) {
        LOG("Ray Reconstruction CreateFeature returned 0x" << std::hex << res << ". Falling back to SuperSampling (Feature 1)..." << std::dec);
        res = pfnCreateFeature(pCL2, NVSDK_NGX_Feature_SuperSampling, pParams, &pHandle);
    }
    pCL2->Close();
    ID3D12CommandList* cl2[] = { pCL2 }; g_pCommandQueue->ExecuteCommandLists(1, cl2); WaitForGPU();
    pCL2->Release(); pCA2->Release();
    if (NVSDK_NGX_FAILED(res)) { LOG("CreateFeature failed: 0x" << std::hex << res); return 1; }
    LOG("NGX Feature created successfully.");

    // Set persistent params
    // Color
    pParams->Set(NVSDK_NGX_Parameter_Color, pColor);
    pParams->Set("DLSSNR.Color", pColor);
    pParams->Set("DLSSD.Color", pColor);
    pParams->Set("DLSSNR.ColorSubrectBaseX", 0);
    pParams->Set("DLSSNR.ColorSubrectBaseY", 0);
    pParams->Set("DLSSNR.ColorSubrectWidth", w);
    pParams->Set("DLSSNR.ColorSubrectHeight", h);

    // Output
    pParams->Set(NVSDK_NGX_Parameter_Output, pOutput);
    pParams->Set("DLSSNR.Output", pOutput);
    pParams->Set("DLSSNR.Backbuffer", pOutput);
    pParams->Set("DLSSD.Output", pOutput);
    pParams->Set("DLSSNR.OutputSubrectBaseX", 0);
    pParams->Set("DLSSNR.OutputSubrectBaseY", 0);
    pParams->Set("DLSSNR.OutputSubrectWidth", out_w);
    pParams->Set("DLSSNR.OutputSubrectHeight", out_h);

    // Depth
    pParams->Set(NVSDK_NGX_Parameter_Depth, pDepth);
    pParams->Set("DLSSNR.Depth", pDepth);
    pParams->Set("DLSSD.Depth", pDepth);
    pParams->Set("DepthInverted", 1);
    pParams->Set("DLSSNR.DepthInverted", 1);
    pParams->Set("DLSSG.DepthInverted", 1);
    pParams->Set("DLSS.Input.Depth.Subrect.Base.X", 0);
    pParams->Set("DLSS.Input.Depth.Subrect.Base.Y", 0);
    pParams->Set("DLSSNR.DepthSubrectBaseX", 0);
    pParams->Set("DLSSNR.DepthSubrectBaseY", 0);
    pParams->Set("DLSSNR.DepthSubrectWidth", w);
    pParams->Set("DLSSNR.DepthSubrectHeight", h);

    // Motion Vectors
    pParams->Set(NVSDK_NGX_Parameter_MotionVectors, pMvec);
    pParams->Set("DLSSNR.MVec", pMvec);
    pParams->Set("DLSSD.MotionVectors", pMvec);
    pParams->Set("DLSS.Input.MV.Subrect.Base.X", 0);
    pParams->Set("DLSS.Input.MV.Subrect.Base.Y", 0);
    pParams->Set("DLSSNR.MVecSubrectBaseX", 0);
    pParams->Set("DLSSNR.MVecSubrectBaseY", 0);
    pParams->Set("DLSSNR.MVecSubrectWidth", out_w);
    pParams->Set("DLSSNR.MVecSubrectHeight", out_h);
    pParams->Set(NVSDK_NGX_Parameter_MV_Scale_X, 1.0f);
    pParams->Set(NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);
    pParams->Set("DLSSNR.MVecScaleX", 1.0f);
    pParams->Set("DLSSNR.MVecScaleY", 1.0f);

    // GBuffer Normals & Roughness
    pParams->Set("Roughness", pNormal);
    pParams->Set("GBuffer.Normals", pNormal);
    pParams->Set("GBuffer.Roughness", pNormal);
    pParams->Set("DLSS.Input.Normals", pNormal);
    pParams->Set("DLSS.Input.Roughness", pNormal);
    pParams->Set("InNormals", pNormal);
    pParams->Set("OutNormals", pNormal);
    pParams->Set("FfxDenoiserNormalsInViewSpace", 1);

    // GBuffer Albedo & Diffuse
    pParams->Set("Albedo", pAlbedo);
    pParams->Set("GBuffer.Albedo", pAlbedo);
    pParams->Set("GBuffer.DiffuseAlbedo", pAlbedo);
    pParams->Set("GBuffer.SpecularAlbedo", pAlbedo);
    pParams->Set("DLSS.Input.DiffuseAlbedo", pAlbedo);
    pParams->Set("DLSS.Input.SpecularAlbedo", pAlbedo);
    pParams->Set("InDiffAlbedo", pAlbedo);
    pParams->Set("InSpecAlbedo", pAlbedo);
    pParams->Set("FfxDenoiserAlbedoFp16", 1);
    pParams->Set("DLSS.Input.Bias.Current.Color.Mask", pAlbedo);

    // Exposure & Sharpening
    pParams->Set(NVSDK_NGX_Parameter_ExposureTexture, pExposure);
    pParams->Set("PreExposure", 1.0f);
    pParams->Set(NVSDK_NGX_Parameter_Sharpness, 0.0f);
    pParams->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
    pParams->Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f);

    // Initial transition for output
    g_pCommandAllocator->Reset(); g_pCommandList->Reset(g_pCommandAllocator, nullptr);
    TransitionResource(g_pCommandList, pOutput, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    g_pCommandList->Close();
    ID3D12CommandList* il[] = { g_pCommandList }; g_pCommandQueue->ExecuteCommandLists(1, il); WaitForGPU();

    // Dummy command list for OptiScaler queue capture
    g_pCommandAllocator->Reset(); g_pCommandList->Reset(g_pCommandAllocator, nullptr);
    g_pCommandList->SetComputeRootSignature(nullptr);
    g_pCommandList->Close();
    ID3D12CommandList* dl[] = { g_pCommandList }; g_pCommandQueue->ExecuteCommandLists(1, dl); WaitForGPU();

    // User-controlled settlement pause before warmup passes
    if (settle_pause_ms > 0) {
        LOG("Applying user settlement pause: " << std::dec << settle_pause_ms << " ms");
        Sleep(settle_pause_ms);
    }

    // =====================================================================
    // WARMUP: Run evaluation passes with initial data to fully settle OptiScaler + AMD-NR
    // =====================================================================
    LOG("=== Running " << std::dec << warmup_passes << " warmup passes (interval: " << warmup_interval_ms << " ms) ===");
    for (int i = 0; i < warmup_passes; ++i) {
        RunOneEvalPass(pParams, pHandle, pfnEval, pOutput, pColor, pDepth, pMvec, pNormal, pAlbedo, pExposure, w, h, out_w, out_h, pSwapChain, i, i == 0);
        if (warmup_interval_ms > 0) {
            Sleep(warmup_interval_ms);
        }
    }
    g_TotalFramesProcessed = warmup_passes;
    LOG("Warmup complete. OptiScaler and AMD-NR are settled and hot.");

    // Run active neural evaluation passes for Frame 0
    LOG("=== Running " << passes << " active neural evaluation passes for Frame 0 ===");
    for (int i = 0; i < passes; ++i) {
        RunOneEvalPass(pParams, pHandle, pfnEval, pOutput, pColor, pDepth, pMvec, pNormal, pAlbedo, pExposure, w, h, out_w, out_h, pSwapChain, g_TotalFramesProcessed + i, false);
    }
    g_TotalFramesProcessed += passes;

    // Download initial output (now fully enhanced by DLSS 5 / AMD-NR)
    g_pCommandAllocator->Reset(); g_pCommandList->Reset(g_pCommandAllocator, nullptr);
    TransitionResource(g_pCommandList, pOutput, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    g_pCommandList->Close();
    ID3D12CommandList* tl[] = { g_pCommandList }; g_pCommandQueue->ExecuteCommandLists(1, tl); WaitForGPU();

    std::vector<uint8_t> outData(out_w * out_h * 8);
    DownloadTexture(pOutput, out_w, out_h, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, outData.data());

    // Write initial output
    { std::ofstream ofs(outputPath, std::ios::binary); ofs.write((char*)outData.data(), outData.size()); }

    // Transition output back to SRV for next frame
    g_pCommandAllocator->Reset(); g_pCommandList->Reset(g_pCommandAllocator, nullptr);
    TransitionResource(g_pCommandList, pOutput, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    g_pCommandList->Close();
    ID3D12CommandList* tl2[] = { g_pCommandList }; g_pCommandQueue->ExecuteCommandLists(1, tl2); WaitForGPU();

    // =====================================================================
    // DAEMON LOOP: Signal ready, then process FRAME commands from stdin
    // =====================================================================
    // Protocol (stdout only):
    //   -> "READY\n"          (bridge is ready for commands)
    //   <- "FRAME color depth mvec output normal albedo\n"  (process a frame)
    //   -> "DONE\n"           (frame processed, output written)
    //   <- "EXIT\n"           (shutdown)
    
    std::cout << "READY" << std::endl; // ONLY protocol message on stdout
    std::cout.flush();

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "EXIT" || line.empty()) break;

        if (line.substr(0, 5) == "FRAME") {
            // Parse: FRAME <color> <depth> <mvec> <output> <normal> <albedo> <eval_passes>
            std::istringstream iss(line);
            std::string cmd, cPath, dPath, mPath, oPath, nPath, aPath;
            int evalPasses = 4;
            iss >> cmd >> cPath >> dPath >> mPath >> oPath >> nPath >> aPath >> evalPasses;

            LOG("Processing FRAME: " << evalPasses << " eval passes");

            // Re-read and re-upload textures
            colorData = ReadFileBin(cPath);
            depthData = ReadFileBin(dPath);
            mvecData  = ReadFileBin(mPath);
            albedoData = ReadFileBin(aPath);
            normalData = ReadFileBin(nPath);

            if (!colorData.empty()) UploadTexture(pColor, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, colorData.data());
            if (!depthData.empty()) UploadTexture(pDepth, w, h, DXGI_FORMAT_R32_FLOAT, 4, depthData.data());
            if (!mvecData.empty())  UploadTexture(pMvec, out_w, out_h, DXGI_FORMAT_R16G16_FLOAT, 4, mvecData.data());
            if (!albedoData.empty()) UploadTexture(pAlbedo, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, albedoData.data());
            if (!normalData.empty()) UploadTexture(pNormal, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, normalData.data());

            // Run evaluation passes (GPU is already warm, no reset needed)
            for (int i = 0; i < evalPasses; ++i) {
                RunOneEvalPass(pParams, pHandle, pfnEval, pOutput, pColor, pDepth, pMvec, pNormal, pAlbedo, pExposure, w, h, out_w, out_h, pSwapChain,
                               g_TotalFramesProcessed + i, false);
            }
            g_TotalFramesProcessed += evalPasses;

            // Download and write output
            g_pCommandAllocator->Reset(); g_pCommandList->Reset(g_pCommandAllocator, nullptr);
            TransitionResource(g_pCommandList, pOutput, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
            g_pCommandList->Close();
            ID3D12CommandList* fl[] = { g_pCommandList }; g_pCommandQueue->ExecuteCommandLists(1, fl); WaitForGPU();

            DownloadTexture(pOutput, out_w, out_h, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, outData.data());
            { std::ofstream ofs(oPath, std::ios::binary); ofs.write((char*)outData.data(), outData.size()); }

            // Transition back for next frame
            g_pCommandAllocator->Reset(); g_pCommandList->Reset(g_pCommandAllocator, nullptr);
            TransitionResource(g_pCommandList, pOutput, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            g_pCommandList->Close();
            ID3D12CommandList* fl2[] = { g_pCommandList }; g_pCommandQueue->ExecuteCommandLists(1, fl2); WaitForGPU();

            LOG("Frame done. Total frames: " << g_TotalFramesProcessed);
            std::cout << "DONE" << std::endl;
            std::cout.flush();
        }
    }

    // Cleanup
    g_WatchdogRunning = false;
    if (hWnd) {
        DestroyWindow(hWnd);
        hWnd = nullptr;
    }
    LOG("Bridge daemon shutting down.");
    return 0;
}
