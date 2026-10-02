#include "ObjectLoading.h"

#include "ObjectLoading.h"

void loadUsdScene(const std::string& filepath,
    std::vector<MeshGeometry>& outMeshes,
    std::vector<SceneInstance>& outInstances,
    Camera& outCamera)
{
    pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(filepath);
    if (!stage) {
        throw std::runtime_error("Failed to open USD stage: " + filepath);
    }

    pxr::UsdGeomXformCache xformCache(pxr::UsdTimeCode::Default());
    bool cameraFound = false;

    for (const auto& prim : stage->Traverse())
    {
        // -------------------------------------------------------------
        // 1. Camera Extraction
        // -------------------------------------------------------------
        if (!cameraFound && prim.IsA<pxr::UsdGeomCamera>()) {
            pxr::UsdGeomCamera usdCam(prim);
            pxr::GfMatrix4d camXform = xformCache.GetLocalToWorldTransform(prim);

            // Eye position in world space
            pxr::GfVec3d eye = camXform.ExtractTranslation();
            outCamera.position = glm::vec3(eye[0], eye[1], eye[2]);

            // USD camera looks down local -Z
            pxr::GfVec3d fwd = camXform.TransformDir(pxr::GfVec3d(0.0, 0.0, -1.0));
            glm::vec3 forwardDir = glm::normalize(glm::vec3(fwd[0], fwd[1], fwd[2]));
            outCamera.target = outCamera.position + forwardDir;

            // USD camera up is local +Y
            pxr::GfVec3d up = camXform.TransformDir(pxr::GfVec3d(0.0, 1.0, 0.0));
            outCamera.up = glm::normalize(glm::vec3(up[0], up[1], up[2]));

            // Vertical FOV calculation
            float focalLength = 50.0f;
            float vertAperture = 24.0f;
            usdCam.GetFocalLengthAttr().Get(&focalLength);
            usdCam.GetVerticalApertureAttr().Get(&vertAperture);
            if (focalLength > 0.0f) {
                outCamera.fovY = glm::degrees(2.0f * std::atan((vertAperture * 0.5f) / focalLength));
            }

            cameraFound = true;
            continue;
        }

        if (!prim.IsA<pxr::UsdGeomMesh>()) continue;

        pxr::UsdGeomMesh usdMesh(prim);

        // -------------------------------------------------------------
        // 2. Fetch Topology & Positions
        // -------------------------------------------------------------
        pxr::VtArray<pxr::GfVec3f> points;
        pxr::VtArray<int> faceVertexCounts;
        pxr::VtArray<int> faceVertexIndices;
        usdMesh.GetPointsAttr().Get(&points);
        usdMesh.GetFaceVertexCountsAttr().Get(&faceVertexCounts);
        usdMesh.GetFaceVertexIndicesAttr().Get(&faceVertexIndices);

        if (points.empty() || faceVertexCounts.empty() || faceVertexIndices.empty()) {
            continue;
        }

        // -------------------------------------------------------------
        // 3. Primvar Queries: Normals & UVs
        // -------------------------------------------------------------
        pxr::UsdGeomPrimvarsAPI primvarsApi(usdMesh);

        // --- Normals Setup ---
        pxr::UsdGeomPrimvar normPrimvar = primvarsApi.GetPrimvar(pxr::TfToken("normals"));
        if (!normPrimvar.IsDefined()) {
            normPrimvar = primvarsApi.GetPrimvar(pxr::TfToken("N"));
        }

        pxr::VtArray<pxr::GfVec3f> normals;
        pxr::VtIntArray normIndices;
        pxr::TfToken normInterp = pxr::UsdGeomTokens->vertex;
        bool normIsIndexed = false;

        if (normPrimvar.IsDefined() && normPrimvar.HasValue()) {
            normPrimvar.Get(&normals);
            normInterp = normPrimvar.GetInterpolation();
            normIsIndexed = normPrimvar.IsIndexed() && normPrimvar.GetIndices(&normIndices);
        }
        else if (usdMesh.GetNormalsAttr().HasAuthoredValue()) {
            usdMesh.GetNormalsAttr().Get(&normals);
            normInterp = usdMesh.GetNormalsInterpolation();
        }

        // --- UVs Setup ---
        pxr::UsdGeomPrimvar uvPrimvar = primvarsApi.GetPrimvar(pxr::TfToken("st"));
        if (!uvPrimvar.IsDefined()) {
            uvPrimvar = primvarsApi.GetPrimvar(pxr::TfToken("uv"));
        }

        pxr::VtArray<pxr::GfVec2f> uvs;
        pxr::VtIntArray uvIndices;
        pxr::TfToken uvInterp = pxr::UsdGeomTokens->vertex;
        bool uvIsIndexed = false;

        if (uvPrimvar.IsDefined() && uvPrimvar.HasValue()) {
            uvPrimvar.Get(&uvs);
            uvInterp = uvPrimvar.GetInterpolation();
            uvIsIndexed = uvPrimvar.IsIndexed() && uvPrimvar.GetIndices(&uvIndices);
        }

        // -------------------------------------------------------------
        // 4. Triangulation & Attribute Assembly
        // -------------------------------------------------------------
        MeshGeometry meshGeom;
        size_t indexOffset = 0;
        size_t faceIndex = 0;

        for (int faceCount : faceVertexCounts)
        {
            for (int i = 1; i < faceCount - 1; ++i)
            {
                size_t cornerIndices[3] = { indexOffset, indexOffset + i, indexOffset + i + 1 };
                int pointIndices[3] = {
                    faceVertexIndices[cornerIndices[0]],
                    faceVertexIndices[cornerIndices[1]],
                    faceVertexIndices[cornerIndices[2]]
                };

                glm::vec3 triPositions[3] = {
                    glm::vec3(points[pointIndices[0]][0], points[pointIndices[0]][1], points[pointIndices[0]][2]),
                    glm::vec3(points[pointIndices[1]][0], points[pointIndices[1]][1], points[pointIndices[1]][2]),
                    glm::vec3(points[pointIndices[2]][0], points[pointIndices[2]][1], points[pointIndices[2]][2])
                };

                // Fallback geometric flat normal if USD has no authored normals
                glm::vec3 geoNormal = glm::cross(triPositions[1] - triPositions[0], triPositions[2] - triPositions[0]);
                float len2 = glm::dot(geoNormal, geoNormal);
                geoNormal = (len2 > 1e-8f) ? glm::normalize(geoNormal) : glm::vec3(0.0f, 1.0f, 0.0f);

                for (int v = 0; v < 3; ++v)
                {
                    size_t corner = cornerIndices[v];
                    int pointIdx = pointIndices[v];

                    // Resolve Normal
                    glm::vec3 norm = geoNormal;
                    if (!normals.empty()) {
                        size_t rawNIdx = pointIdx;
                        if (normInterp == pxr::UsdGeomTokens->faceVarying) {
                            rawNIdx = corner;
                        }
                        else if (normInterp == pxr::UsdGeomTokens->uniform) {
                            rawNIdx = faceIndex;
                        }

                        size_t finalNIdx = (normIsIndexed && rawNIdx < normIndices.size()) ? normIndices[rawNIdx] : rawNIdx;
                        if (finalNIdx < normals.size()) {
                            norm = glm::vec3(normals[finalNIdx][0], normals[finalNIdx][1], normals[finalNIdx][2]);
                        }
                    }

                    // Resolve UV
                    glm::vec2 uv(0.0f);
                    if (!uvs.empty()) {
                        size_t rawUvIdx = pointIdx;
                        if (uvInterp == pxr::UsdGeomTokens->faceVarying) {
                            rawUvIdx = corner;
                        }
                        else if (uvInterp == pxr::UsdGeomTokens->uniform) {
                            rawUvIdx = faceIndex;
                        }

                        size_t finalUvIdx = (uvIsIndexed && rawUvIdx < uvIndices.size()) ? uvIndices[rawUvIdx] : rawUvIdx;
                        if (finalUvIdx < uvs.size()) {
                            uv = glm::vec2(uvs[finalUvIdx][0], uvs[finalUvIdx][1]);
                        }
                    }

                    meshGeom.vertices.push_back(triPositions[v]);
                    meshGeom.attributes.push_back({ norm, 0.0f, uv, glm::vec2(0.0f) });
                    meshGeom.indices.push_back(static_cast<uint32_t>(meshGeom.vertices.size() - 1));
                }
            }

            indexOffset += faceCount;
            faceIndex++;
        }

        // -------------------------------------------------------------
        // 5. Transform Matrix & Instance Registry
        // -------------------------------------------------------------
        pxr::GfMatrix4d worldTransform = xformCache.GetLocalToWorldTransform(prim);
        glm::mat4 transform(1.0f);
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 4; ++r) {
                transform[c][r] = static_cast<float>(worldTransform[c][r]);
            }
        }

        uint32_t meshIndex = static_cast<uint32_t>(outMeshes.size());
        outMeshes.push_back(std::move(meshGeom));

        outInstances.push_back(SceneInstance
            {
                .meshIndex = meshIndex,
                .transform = transform,
                .customInstanceId = static_cast<uint32_t>(outInstances.size()),
                .materialId = 0
            });
    }
}

namespace mx = MaterialX;

WavefrontData::GpuOpenPbrMaterial loadOpenPbrFromMtlx(const std::string& mtlxPath)
{
    mx::DocumentPtr doc = mx::createDocument();
    mx::readFromXmlFile(doc, mtlxPath);

    WavefrontData::GpuOpenPbrMaterial mat{};

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
            mat.baseColorCoeffs = glm::vec4(getColor3("base_color", glm::vec3(0.8f)), 1.0f);
            mat.baseRoughness = getFloat("base_roughness", 0.3f);
            mat.baseMetalness = getFloat("base_metalness", 0.0f);

            // Specular & Transmission
            mat.specularWeight = getFloat("specular_weight", 1.0f);
            mat.specularRoughness = getFloat("specular_roughness", 0.3f);
            mat.specularIor = getFloat("specular_ior", 1.5f);
            mat.transmissionWeight = getFloat("transmission_weight", 0.0f);
            mat.transmissionColorCoeffs = glm::vec4(getColor3("transmission_color", glm::vec3(1.0f)), 1.0f);

            // Coat
            mat.coatWeight = getFloat("coat_weight", 0.0f);
            mat.coatRoughness = getFloat("coat_roughness", 0.05f);
            mat.coatIor = getFloat("coat_ior", 1.6f);

            // Emission
            glm::vec3 emissionCol = getColor3("emission_color", glm::vec3(0.0f));
            float emissionLum = getFloat("emission_luminance", 0.0f);
            mat.emissionAndLuminance = glm::vec4(emissionCol, emissionLum);
            break;
        }
    }
    return mat;
}