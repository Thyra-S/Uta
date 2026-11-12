#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <unordered_map>

#include "SpectralWavefrontPt.h"

#pragma warning(push)
#pragma warning(disable: 4244 4305)
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/imaging/hio/image.h>
#pragma warning(pop)

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/XmlIo.h>

struct CpuTextureData {
    std::string filepath;
    uint32_t width = 0;
    uint32_t height = 0;
    bool isSrgb = true; // true =VK_FORMAT_R8G8B8A8_SRGB, false = VK_FORMAT_R8G8B8A8_UNORM
    std::vector<uint8_t> pixels; // RGBA8 tightly packed
};

// Loads meshes, instances, bound USD/MaterialX materials, textures, and optional USD camera
void loadUsdScene(const std::string& filepath,
    std::vector<MeshGeometry>& outMeshes,
    std::vector<SceneInstance>& outInstances,
    std::vector<WavefrontData::GpuOpenPbrMaterial>& outMaterials,
    std::vector<CpuTextureData>& outTextures,
    Camera& outCamera);

// Standalone .mtlx loader (also used internally if a USD material references a .mtlx asset)
WavefrontData::GpuOpenPbrMaterial loadOpenPbrFromMtlx(
    const std::string& mtlxPath,
    std::vector<CpuTextureData>& outTextures,
    std::unordered_map<std::string, uint32_t>& textureCache);

// Overload if you just want to load a single .mtlx without textures
WavefrontData::GpuOpenPbrMaterial loadOpenPbrFromMtlx(const std::string& mtlxPath);