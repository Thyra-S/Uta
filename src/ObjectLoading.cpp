#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <MaterialXCore/Document.h>
#include <MaterialXFormat/XmlIo.h>
#include "SpectralWavefrontPt.h"

void loadUsdScene(const std::string& filepath,
    std::vector<MeshGeometry>& outMeshes,
    std::vector<SceneInstance>& outInstances)
{
    pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(filepath);
    if (!stage) {
        throw std::runtime_error("Failed to open USD stage: " + filepath);
    }

    pxr::UsdGeomXformCache xformCache(pxr::UsdTimeCode::Default());

    for (const auto& prim : stage->Traverse()) {
        if (!prim.IsA<pxr::UsdGeomMesh>()) continue;

        pxr::UsdGeomMesh usdMesh(prim);

        // 1. Fetch Topology & Vertices
        pxr::VtArray<pxr::GfVec3f> points;
        pxr::VtArray<int> faceVertexCounts;
        pxr::VtArray<int> faceVertexIndices;
        usdMesh.GetPointsAttr().Get(&points);
        usdMesh.GetFaceVertexCountsAttr().Get(&faceVertexCounts);
        usdMesh.GetFaceVertexIndicesAttr().Get(&faceVertexIndices);

        // 2. Fetch Primvars (Normals and UVs)
        pxr::VtArray<pxr::GfVec3f> normals;
        usdMesh.GetNormalsAttr().Get(&normals);

        pxr::UsdGeomPrimvar uvPrimvar = usdMesh.GetPrimvar(pxr::TfToken("st"));
        if (!uvPrimvar.IsDefined()) {
            uvPrimvar = usdMesh.GetPrimvar(pxr::TfToken("uv"));
        }
        pxr::VtArray<pxr::GfVec2f> uvs;
        if (uvPrimvar.IsDefined()) {
            uvPrimvar.Get(&uvs);
        }

        MeshGeometry meshGeom;
        size_t indexOffset = 0;

        // 3. Triangulate polygons into meshGeom
        for (int faceCount : faceVertexCounts) {
            for (int i = 1; i < faceCount - 1; ++i) {
                int idx0 = faceVertexIndices[indexOffset];
                int idx1 = faceVertexIndices[indexOffset + i];
                int idx2 = faceVertexIndices[indexOffset + i + 1];

                auto addVertex = [&](int srcIdx) -> uint32_t {
                    glm::vec3 pos(points[srcIdx][0], points[srcIdx][1], points[srcIdx][2]);
                    glm::vec3 norm = normals.empty() ? glm::vec3(0, 1, 0) :
                        glm::vec3(normals[srcIdx][0], normals[srcIdx][1], normals[srcIdx][2]);
                    glm::vec2 uv = uvs.empty() ? glm::vec2(0.0f) :
                        glm::vec2(uvs[srcIdx][0], uvs[srcIdx][1]);

                    meshGeom.vertices.push_back(pos);
                    meshGeom.attributes.push_back({ norm, uv });
                    return static_cast<uint32_t>(meshGeom.vertices.size() - 1);
                    };

                meshGeom.indices.push_back(addVertex(idx0));
                meshGeom.indices.push_back(addVertex(idx1));
                meshGeom.indices.push_back(addVertex(idx2));
            }
            indexOffset += faceCount;
        }

        // 4. Transform Matrix
        pxr::GfMatrix4d worldTransform = xformCache.GetLocalToWorldTransform(prim);
        glm::mat4 transform(1.0f);
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                transform[c][r] = static_cast<float>(worldTransform[r][c]);
            }
        }

        uint32_t meshIndex = static_cast<uint32_t>(outMeshes.size());
        outMeshes.push_back(std::move(meshGeom));

        outInstances.push_back(SceneInstance
            {
            .meshIndex = meshIndex,
            .transform = transform,
            .customInstanceId = static_cast<uint32_t>(outInstances.size()),
            .materialId = 0 // Resolved from UsdShadeMaterialBindingAPI
            });
    }
}

namespace mx = MaterialX;

WavefrontData::GpuOpenPbrMaterial loadOpenPbrFromMtlx(const std::string& mtlxPath)
{
    mx::DocumentPtr doc = mx::createDocument();
    mx::readFromXmlFile(doc, mtlxPath);

    WavefrontData::GpuOpenPbrMaterial mat{};

    // Find the open_pbr_surface shader node
    for (const auto& node : doc->getNodes()) {
        if (node->getCategory() == "open_pbr_surface" ||
            node->getType() == "surfaceshader")
        {
            auto getFloat = [&](const std::string& name, float defaultVal) -> float {
                auto input = node->getInput(name);
                return input ? input->getValue()->asA<float>() : defaultVal;
                };

            auto getColor3 = [&](const std::string& name, glm::vec3 defaultVal) -> glm::vec3 {
                auto input = node->getInput(name);
                if (!input) return defaultVal;
                auto col = input->getValue()->asA<mx::Color3>();
                return glm::vec3(col[0], col[1], col[2]);
                };

            // Base
            mat.baseWeight = getFloat("base_weight", 1.0f);
            mat.baseColor = getColor3("base_color", glm::vec3(0.8f));
            mat.baseRoughness = getFloat("base_roughness", 0.3f);
            mat.baseMetalness = getFloat("base_metalness", 0.0f);

            // Specular & Transmission
            mat.specularWeight = getFloat("specular_weight", 1.0f);
            mat.specularRoughness = getFloat("specular_roughness", 0.3f);
            mat.specularIor = getFloat("specular_ior", 1.5f);
            mat.transmissionWeight = getFloat("transmission_weight", 0.0f);
            mat.transmissionColor = getColor3("transmission_color", glm::vec3(1.0f));

            // Coat
            mat.coatWeight = getFloat("coat_weight", 0.0f);
            mat.coatRoughness = getFloat("coat_roughness", 0.05f);
            mat.coatIor = getFloat("coat_ior", 1.6f);

            // Emission
            mat.emissionLuminance = getFloat("emission_luminance", 0.0f);
            mat.emissionColor = getColor3("emission_color", glm::vec3(0.0f));
            break;
        }
    }
    return mat;
}
