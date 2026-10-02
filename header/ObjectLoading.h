#pragma once

#include <string>
#include <vector>
#include "SpectralWavefrontPt.h"
#pragma warning(push)
#pragma warning(disable: 4244 4305)
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#pragma warning(pop)

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/XmlIo.h>
#include "SpectralWavefrontPt.h"

// Loads meshes, instances, and optional USD camera properties
void loadUsdScene(const std::string& filepath,
    std::vector<MeshGeometry>& outMeshes,
    std::vector<SceneInstance>& outInstances,
    Camera& outCamera);

WavefrontData::GpuOpenPbrMaterial loadOpenPbrFromMtlx(const std::string& mtlxPath);