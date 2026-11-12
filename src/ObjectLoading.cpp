#include "ObjectLoading.h"

namespace mx = MaterialX;

// -------------------------------------------------------------------------
// Helper: Load & Deduplicate Texture via OpenUSD HioImage
// -------------------------------------------------------------------------
static uint32_t registerOrLoadTexture(
    const std::string& resolvedPath,
    bool isSrgb,
    std::vector<CpuTextureData>& outTextures,
    std::unordered_map<std::string, uint32_t>& textureCache)
{
    if (resolvedPath.empty()) return ~0u;

    auto it = textureCache.find(resolvedPath);
    if (it != textureCache.end()) {
        return it->second;
    }

    pxr::HioImageSharedPtr img = pxr::HioImage::OpenForReading(resolvedPath);
    if (!img) {
        std::cerr << "[Warning] Failed to open texture: " << resolvedPath << "\n";
        return ~0u;
    }

    CpuTextureData tex{};
    tex.filepath = resolvedPath;
    tex.width = static_cast<uint32_t>(img->GetWidth());
    tex.height = static_cast<uint32_t>(img->GetHeight());
    tex.isSrgb = isSrgb;
    tex.pixels.resize(static_cast<size_t>(tex.width) * tex.height * 4);

    pxr::HioImage::StorageSpec spec{};
    spec.width = img->GetWidth();
    spec.height = img->GetHeight();
    spec.depth = 1;
    spec.format = pxr::HioFormatUNorm8Vec4;
    spec.flipped = true; // Flip USD bottom-left UV origin to match Vulkan top-left
    spec.data = tex.pixels.data();

    if (!img->Read(spec)) {
        std::cerr << "[Warning] Failed to decode texture pixels: " << resolvedPath << "\n";
        return ~0u;
    }

    uint32_t texId = static_cast<uint32_t>(outTextures.size());
    outTextures.push_back(std::move(tex));
    textureCache[resolvedPath] = texId;
    return texId;
}

// -------------------------------------------------------------------------
// MaterialX (.mtlx) OpenPBR Loader (Full Struct + Textures)
// -------------------------------------------------------------------------
WavefrontData::GpuOpenPbrMaterial loadOpenPbrFromMtlx(
    const std::string& mtlxPath,
    std::vector<CpuTextureData>& outTextures,
    std::unordered_map<std::string, uint32_t>& textureCache)
{
    mx::DocumentPtr doc = mx::createDocument();
    mx::readFromXmlFile(doc, mtlxPath);

    std::filesystem::path baseDir = std::filesystem::path(mtlxPath).parent_path();
    WavefrontData::GpuOpenPbrMaterial mat{};

    for (const auto& node : doc->getNodes()) {
        if (node->getCategory() == "open_pbr_surface" ||
            node->getType() == "surfaceshader")
        {
            auto getFloat = [&](const std::string& name, float defaultVal) -> float {
                auto input = node->getInput(name);
                return (input && input->getValue()) ? input->getValue()->asA<float>() : defaultVal;
                };

            auto getBool = [&](const std::string& name, bool defaultVal) -> bool {
                auto input = node->getInput(name);
                return (input && input->getValue()) ? input->getValue()->asA<bool>() : defaultVal;
                };

            auto getColor3 = [&](const std::string& name, glm::vec3 defaultVal) -> glm::vec3 {
                auto input = node->getInput(name);
                if (!input || !input->getValue()) return defaultVal;
                auto col = input->getValue()->asA<mx::Color3>();
                return glm::vec3(col[0], col[1], col[2]);
                };

            auto getTexId = [&](const std::string& name, bool isSrgb) -> uint32_t {
                mx::InputPtr input = node->getInput(name);
                if (!input) return ~0u;

                mx::NodePtr upstream = input->getConnectedNode();
                if (!upstream) {
                    if (mx::OutputPtr graphOut = input->getConnectedOutput()) {
                        upstream = graphOut->getConnectedNode();
                    }
                }

                // Walk upstream past intermediate nodes (e.g., <normalmap>, <extract>, <multiply>)
                int hops = 0;
                while (upstream && !upstream->getInput("file") && hops++ < 8) {
                    mx::InputPtr nextIn = upstream->getInput("in");
                    if (!nextIn) nextIn = upstream->getInput("in1");
                    upstream = nextIn ? nextIn->getConnectedNode() : nullptr;
                }

                if (upstream) {
                    if (mx::InputPtr fileInput = upstream->getInput("file")) {
                        std::string relPath = fileInput->getValueString();
                        if (!relPath.empty()) {
                            std::string fullPath = (baseDir / relPath).lexically_normal().string();
                            return registerOrLoadTexture(fullPath, isSrgb, outTextures, textureCache);
                        }
                    }
                }
                return ~0u;
                };

            // 1. Base Lobe

            // 7. Texture Map Bindings
            mat.baseColorTexId = getTexId("base_color", true);
            mat.specularRoughnessTexId = getTexId("specular_roughness", false);
            if (mat.specularRoughnessTexId == ~0u) {
                mat.specularRoughnessTexId = getTexId("base_roughness", false);
            }
            mat.coatRoughnessTexId = getTexId("coat_roughness", false);
            mat.normalTexId = getTexId("geometry_normal", false);

            glm::vec3 defaultBaseCol = (mat.baseColorTexId != ~0u) ? glm::vec3(1.0f) : glm::vec3(0.8f);
            mat.baseColorCoeffs = glm::vec4(getColor3("base_color", defaultBaseCol), 1.0f); mat.baseWeight = getFloat("base_weight", 1.0f);
            mat.baseRoughness = getFloat("base_roughness", 0.3f);
            mat.baseMetalness = getFloat("base_metalness", 0.0f);
            mat.baseDiffuseRoughness = getFloat("base_diffuse_roughness", 0.0f);

            // 2. Specular Lobe
            mat.specularColorCoeffs = glm::vec4(getColor3("specular_color", glm::vec3(1.0f)), 1.0f);
            mat.specularWeight = getFloat("specular_weight", 1.0f);
            mat.specularRoughness = getFloat("specular_roughness", 0.3f);
            mat.specularIor = getFloat("specular_ior", 1.5f);
            mat.specularRoughnessAnisotropy = getFloat("specular_roughness_anisotropy", 0.0f);

            // 3. Transmission Lobe
            mat.transmissionColorCoeffs = glm::vec4(getColor3("transmission_color", glm::vec3(1.0f)), 1.0f);
            glm::vec3 transScatter = getColor3("transmission_scatter", glm::vec3(0.0f));
            float abbeNumber = getFloat("transmission_dispersion_abbe_number", 0.0f);
            float dispScale = getFloat("transmission_dispersion_scale", 0.0f);
            float effectiveDispersion = (dispScale > 0.0f && abbeNumber > 0.0f) ? (abbeNumber / dispScale) : 0.0f;
            mat.transmissionScatterAndDispersion = glm::vec4(transScatter, effectiveDispersion);
            mat.transmissionWeight = getFloat("transmission_weight", 0.0f);
            mat.transmissionDepth = getFloat("transmission_depth", 0.0f);

            // 4. Subsurface Lobe
            glm::vec3 sssColor = getColor3("subsurface_color", glm::vec3(0.8f));
            float sssWeight = getFloat("subsurface_weight", 0.0f);
            mat.subsurfaceColorAndWeight = glm::vec4(sssColor, sssWeight);
            glm::vec3 sssRadius = getColor3("subsurface_radius", glm::vec3(1.0f));
            float sssScale = getFloat("subsurface_radius_scale", 1.0f);
            mat.subsurfaceRadiusAndScale = glm::vec4(sssRadius, sssScale);

            // 5. Coat Lobe
            glm::vec3 coatColor = getColor3("coat_color", glm::vec3(1.0f));
            float coatDarkening = getFloat("coat_darkening", 1.0f);
            mat.coatColorAndDarkening = glm::vec4(coatColor, coatDarkening);
            mat.coatWeight = getFloat("coat_weight", 0.0f);
            mat.coatRoughness = getFloat("coat_roughness", 0.05f);
            mat.coatIor = getFloat("coat_ior", 1.6f);

            // 6. Emission & Geometry
            glm::vec3 emissionCol = getColor3("emission_color", glm::vec3(0.0f));
            float emissionLum = getFloat("emission_luminance", 0.0f);
            mat.emissionAndLuminance = glm::vec4(emissionCol, emissionLum);
            mat.thinWalled = getBool("geometry_thin_walled", false) ? 1u : 0u;

            break;
        }
    }
    return mat;
}

WavefrontData::GpuOpenPbrMaterial loadOpenPbrFromMtlx(const std::string& mtlxPath)
{
    std::vector<CpuTextureData> dummyTextures;
    std::unordered_map<std::string, uint32_t> dummyCache;
    return loadOpenPbrFromMtlx(mtlxPath, dummyTextures, dummyCache);
}

// -------------------------------------------------------------------------
// USD Material & Texture Extractor
// -------------------------------------------------------------------------
template <typename T>
static bool readUsdInputOrTexture(
    const pxr::UsdShadeShader& shader,
    const char* inputName,
    bool isSrgb,
    T& outValue,
    uint32_t& outTexId,
    std::vector<CpuTextureData>& outTextures,
    std::unordered_map<std::string, uint32_t>& textureCache)
{
    outTexId = ~0u;
    pxr::UsdShadeInput input = shader.GetInput(pxr::TfToken(inputName));
    if (!input) return false;

    // 1. Check if connected to an upstream texture node (UsdUVTexture / ND_image)
    pxr::UsdShadeConnectableAPI source;
    pxr::TfToken sourceName;
    pxr::UsdShadeAttributeType sourceType;

    if (pxr::UsdShadeConnectableAPI::GetConnectedSource(input, &source, &sourceName, &sourceType)) {
        pxr::UsdShadeShader upstreamShader(source.GetPrim());
        // Walk up 1 level if connected via a normalmap or scale node
        if (upstreamShader && !upstreamShader.GetInput(pxr::TfToken("file"))) {
            pxr::UsdShadeInput inAttr = upstreamShader.GetInput(pxr::TfToken("in"));
            if (inAttr && pxr::UsdShadeConnectableAPI::GetConnectedSource(inAttr, &source, &sourceName, &sourceType)) {
                upstreamShader = pxr::UsdShadeShader(source.GetPrim());
            }
        }
        if (upstreamShader) {
            pxr::UsdShadeInput fileInput = upstreamShader.GetInput(pxr::TfToken("file"));
            if (fileInput) {
                pxr::SdfAssetPath assetPath;
                if (fileInput.Get(&assetPath)) {
                    std::string resolved = assetPath.GetResolvedPath();
                    if (resolved.empty()) resolved = assetPath.GetAssetPath();
                    outTexId = registerOrLoadTexture(resolved, isSrgb, outTextures, textureCache);
                    return true;
                }
            }
        }
    }

    // 2. Otherwise read constant value
    return input.Get(&outValue);
}

static uint32_t resolveUsdMaterial(
    const pxr::UsdShadeMaterial& usdMat,
    std::vector<WavefrontData::GpuOpenPbrMaterial>& outMaterials,
    std::unordered_map<std::string, uint32_t>& materialCache,
    std::vector<CpuTextureData>& outTextures,
    std::unordered_map<std::string, uint32_t>& textureCache)
{
    std::string matKey = usdMat.GetPath().GetString();
    auto it = materialCache.find(matKey);
    if (it != materialCache.end()) {
        return it->second;
    }

    WavefrontData::GpuOpenPbrMaterial mat{};

    // Prefer MaterialX render context if present, fallback to universal surface output
    pxr::UsdShadeShader surfaceShader = usdMat.ComputeSurfaceSource({ pxr::TfToken("mtlx") });
    if (!surfaceShader) {
        surfaceShader = usdMat.ComputeSurfaceSource();
    }

    if (surfaceShader) {
        // Case 1: External .mtlx asset referenced directly by the USD shader
        pxr::SdfAssetPath sourceAsset;
        if (surfaceShader.GetSourceAsset(&sourceAsset, pxr::TfToken("mtlx"))) {
            std::string resolvedMtlx = sourceAsset.GetResolvedPath();
            if (!resolvedMtlx.empty()) {
                mat = loadOpenPbrFromMtlx(resolvedMtlx, outTextures, textureCache);
            }
        }
        else {
            pxr::TfToken shaderId;
            surfaceShader.GetShaderId(&shaderId);

            pxr::GfVec3f colVal(0.0f);
            float fltVal = 0.0f;
            bool boolVal = false;
            uint32_t texId = ~0u;

            // Case 2: OpenPBR authored directly in USD (UsdMtlx)
            if (shaderId == pxr::TfToken("ND_open_pbr_surface_surfaceshader")) {
                if (readUsdInputOrTexture(surfaceShader, "base_color", true, colVal, texId, outTextures, textureCache)) {
                    if (texId != ~0u) mat.baseColorTexId = texId;
                    else mat.baseColorCoeffs = glm::vec4(colVal[0], colVal[1], colVal[2], 1.0f);
                }
                if (readUsdInputOrTexture(surfaceShader, "base_weight", false, fltVal, texId, outTextures, textureCache) && texId == ~0u)
                    mat.baseWeight = fltVal;
                if (readUsdInputOrTexture(surfaceShader, "base_roughness", false, fltVal, texId, outTextures, textureCache) && texId == ~0u)
                    mat.baseRoughness = fltVal;
                if (readUsdInputOrTexture(surfaceShader, "base_metalness", false, fltVal, texId, outTextures, textureCache) && texId == ~0u)
                    mat.baseMetalness = fltVal;
                if (readUsdInputOrTexture(surfaceShader, "specular_roughness", false, fltVal, texId, outTextures, textureCache)) {
                    if (texId != ~0u) mat.specularRoughnessTexId = texId;
                    else mat.specularRoughness = fltVal;
                }
                if (readUsdInputOrTexture(surfaceShader, "specular_ior", false, fltVal, texId, outTextures, textureCache) && texId == ~0u)
                    mat.specularIor = fltVal;
                if (readUsdInputOrTexture(surfaceShader, "transmission_weight", false, fltVal, texId, outTextures, textureCache) && texId == ~0u)
                    mat.transmissionWeight = fltVal;
                if (readUsdInputOrTexture(surfaceShader, "coat_weight", false, fltVal, texId, outTextures, textureCache) && texId == ~0u)
                    mat.coatWeight = fltVal;
                if (readUsdInputOrTexture(surfaceShader, "coat_roughness", false, fltVal, texId, outTextures, textureCache)) {
                    if (texId != ~0u) mat.coatRoughnessTexId = texId;
                    else mat.coatRoughness = fltVal;
                }
                if (readUsdInputOrTexture(surfaceShader, "geometry_normal", false, colVal, texId, outTextures, textureCache) && texId != ~0u)
                    mat.normalTexId = texId;
                if (readUsdInputOrTexture(surfaceShader, "geometry_thin_walled", false, boolVal, texId, outTextures, textureCache) && texId == ~0u)
                    mat.thinWalled = boolVal ? 1u : 0u;
            }
            // Case 3: UsdPreviewSurface mapped to OpenPBR
            else {
                if (readUsdInputOrTexture(surfaceShader, "diffuseColor", true, colVal, texId, outTextures, textureCache)) {
                    if (texId != ~0u) mat.baseColorTexId = texId;
                    else mat.baseColorCoeffs = glm::vec4(colVal[0], colVal[1], colVal[2], 1.0f);
                }
                if (readUsdInputOrTexture(surfaceShader, "roughness", false, fltVal, texId, outTextures, textureCache)) {
                    if (texId != ~0u) mat.specularRoughnessTexId = texId;
                    else {
                        mat.baseRoughness = fltVal;
                        mat.specularRoughness = fltVal;
                    }
                }
                if (readUsdInputOrTexture(surfaceShader, "metallic", false, fltVal, texId, outTextures, textureCache) && texId == ~0u)
                    mat.baseMetalness = fltVal;
                if (readUsdInputOrTexture(surfaceShader, "ior", false, fltVal, texId, outTextures, textureCache) && texId == ~0u)
                    mat.specularIor = fltVal;
                if (readUsdInputOrTexture(surfaceShader, "opacity", false, fltVal, texId, outTextures, textureCache) && texId == ~0u)
                    mat.transmissionWeight = 1.0f - fltVal;
                if (readUsdInputOrTexture(surfaceShader, "clearcoat", false, fltVal, texId, outTextures, textureCache) && texId == ~0u)
                    mat.coatWeight = fltVal;
                if (readUsdInputOrTexture(surfaceShader, "clearcoatRoughness", false, fltVal, texId, outTextures, textureCache)) {
                    if (texId != ~0u) mat.coatRoughnessTexId = texId;
                    else mat.coatRoughness = fltVal;
                }
                if (readUsdInputOrTexture(surfaceShader, "normal", false, colVal, texId, outTextures, textureCache) && texId != ~0u)
                    mat.normalTexId = texId;
                if (readUsdInputOrTexture(surfaceShader, "emissiveColor", true, colVal, texId, outTextures, textureCache) && texId == ~0u) {
                    float maxC = std::max({ colVal[0], colVal[1], colVal[2] });
                    mat.emissionAndLuminance = glm::vec4(colVal[0], colVal[1], colVal[2], maxC > 0.0f ? 1.0f : 0.0f);
                }
            }
        }
    }

    uint32_t newMatId = static_cast<uint32_t>(outMaterials.size());
    outMaterials.push_back(mat);
    materialCache[matKey] = newMatId;
    return newMatId;
}

// -------------------------------------------------------------------------
// Main USD Scene Loader
// -------------------------------------------------------------------------
void loadUsdScene(const std::string& filepath,
    std::vector<MeshGeometry>& outMeshes,
    std::vector<SceneInstance>& outInstances,
    std::vector<WavefrontData::GpuOpenPbrMaterial>& outMaterials,
    std::vector<CpuTextureData>& outTextures,
    Camera& outCamera)
{
    pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(filepath);
    if (!stage) {
        throw std::runtime_error("Failed to open USD stage: " + filepath);
    }

    // Ensure index 0 is always a valid fallback default material
    if (outMaterials.empty()) {
        WavefrontData::GpuOpenPbrMaterial defaultMat{};
        defaultMat.baseColorCoeffs = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
        outMaterials.push_back(defaultMat);
    }

    std::unordered_map<std::string, uint32_t> materialCache;
    std::unordered_map<std::string, uint32_t> textureCache;
    for (uint32_t i = 0; i < static_cast<uint32_t>(outTextures.size()); ++i) {
        textureCache[outTextures[i].filepath] = i;
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

            pxr::GfVec3d eye = camXform.ExtractTranslation();
            outCamera.position = glm::vec3(eye[0], eye[1], eye[2]);

            pxr::GfVec3d fwd = camXform.TransformDir(pxr::GfVec3d(0.0, 0.0, -1.0));
            glm::vec3 forwardDir = glm::normalize(glm::vec3(fwd[0], fwd[1], fwd[2]));
            outCamera.target = outCamera.position + forwardDir;

            pxr::GfVec3d up = camXform.TransformDir(pxr::GfVec3d(0.0, 1.0, 0.0));
            outCamera.up = glm::normalize(glm::vec3(up[0], up[1], up[2]));

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

                glm::vec3 geoNormal = glm::cross(triPositions[1] - triPositions[0], triPositions[2] - triPositions[0]);
                float len2 = glm::dot(geoNormal, geoNormal);
                geoNormal = (len2 > 1e-8f) ? glm::normalize(geoNormal) : glm::vec3(0.0f, 1.0f, 0.0f);

                for (int v = 0; v < 3; ++v)
                {
                    size_t corner = cornerIndices[v];
                    int pointIdx = pointIndices[v];

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
        // 5. Material Binding, Transform Matrix & Instance Registry
        // -------------------------------------------------------------
        uint32_t matId = 0;
        pxr::UsdShadeMaterialBindingAPI bindingApi(prim);
        if (pxr::UsdShadeMaterial boundMat = bindingApi.ComputeBoundMaterial()) {
            matId = resolveUsdMaterial(boundMat, outMaterials, materialCache, outTextures, textureCache);
        }

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
                .materialId = matId
            });
    }
}