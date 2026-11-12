// Shade.slang
import Common;

[[vk::push_constant]]
ConstantBuffer<SceneParams> params;

// Descriptor Set 0 Bindings
[[vk::binding(1, 0)]] RWTexture2D<float4>            g_AccumulationImage;
[[vk::binding(2, 0)]] RWStructuredBuffer<RayPayload> g_RayQueue;
[[vk::binding(3, 0)]] RWStructuredBuffer<HitPayload> g_HitQueue;
[[vk::binding(5, 0)]] Sampler2D g_Textures[];

// Decorrelated PCG hash state initializer per pixel, sample, and bounce
uint initRng(uint pixelIndex, uint frameCounter, uint bounce)
{
    uint state = pixelIndex + (frameCounter * 719393u) + (bounce * 17971u);
    return pcg_hash(state);
}

// Physically-based daylight sky dome for miss rays
float3 getEnvironmentRadiance(float3 dir)
{
    float t = clamp(dir.y, 0.0f, 1.0f);

    float3 zenith  = float3(0.65f);
    float3 horizon = float3(0.85f);
    float3 ground  = float3(0.22f);

    float3 sky = (dir.y >= 0.0f)
        ? lerp(horizon, zenith, sqrt(t))
        : lerp(horizon, ground, clamp(-dir.y * 4.0f, 0.0f, 1.0f));

    float3 sunDir = normalize(float3(0.5f, 0.8f, 0.3f));
    if (dot(dir, sunDir) > 0.92f)
    {
        sky += float3(3.0f);
    }
    return sky;
}

// Samples an isotropic GGX microfacet half-vector and evaluates the Smith G2 BRDF weight:
// weight = (G * VdotH) / (NdotV * NdotH). Returns false if the reflected ray falls below the surface.
bool sampleGGXMicrofacet(
    float2     u,
    float      alpha,
    float3     V,
    float3     normal,
    float      cosThetaV,
    out float3 nextDir,
    out float  VdotH,
    out float  weight)
{
    float a2 = alpha * alpha;
    float phi = 2.0f * 3.1415926535f * u.x;
    float cosThetaH = sqrt((1.0f - u.y) / max(1e-7f, 1.0f + (a2 - 1.0f) * u.y));
    float sinThetaH = sqrt(max(0.0f, 1.0f - cosThetaH * cosThetaH));

    float3 H_local = float3(sinThetaH * cos(phi), sinThetaH * sin(phi), cosThetaH);
    float3 H = normalize(toWorld(H_local, normal));
    nextDir = reflect(-V, H);

    float NdotL = dot(normal, nextDir);
    if (NdotL <= 0.0f)
    {
        VdotH  = 0.0f;
        weight = 0.0f;
        return false;
    }

    VdotH = max(1e-4f, dot(V, H));
    float NdotV = max(1e-4f, cosThetaV);
    float NdotH = max(1e-4f, cosThetaH);

    float G1_V = (2.0f * NdotV) / (NdotV + sqrt(a2 + (1.0f - a2) * NdotV * NdotV));
    float G1_L = (2.0f * NdotL) / (NdotL + sqrt(a2 + (1.0f - a2) * NdotL * NdotL));

    weight = (G1_V * G1_L * VdotH) / (NdotV * NdotH);
    return true;
}

void terminateRay(uint rayIndex, inout RayPayload ray)
{
    ray.isTerminated = 1;
    g_RayQueue[rayIndex] = ray;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixelCoord = dispatchThreadID.xy;
    if (pixelCoord.x >= params.width || pixelCoord.y >= params.height)
        return;

    uint rayIndex = pixelCoord.y * params.width + pixelCoord.x;

    RayPayload ray = g_RayQueue[rayIndex];
    if (ray.isTerminated != 0)
        return;

    HitPayload hit        = g_HitQueue[rayIndex];
    uint       seed       = initRng(rayIndex, params.frameCounter, params.currentBounce);

    // 1. Ray Miss (Environment Sky)
    if (hit.hitStatus == 0)
    {
        float3 skyRgb       = getEnvironmentRadiance(ray.direction);
        float4 skySpectrum  = rgbToSpectrum4(skyRgb, ray.wavelengths);
        float4 missRadiance = ray.throughput * skySpectrum;
        float3 missRgb      = spectral4ToLinearRgb(missRadiance, ray.wavelengths);

        g_AccumulationImage[pixelCoord] += float4(missRgb, 0.0f);
        terminateRay(rayIndex, ray);
        return;
    }

    // 2. Geometry & Material Resolution
    GpuMeshDesc*        meshDescs = (GpuMeshDesc*)params.vertexBufferAddress;
    GpuOpenPbrMaterial* materials = (GpuOpenPbrMaterial*)params.materialBufferAddress;

    GpuMeshDesc        mesh = meshDescs[hit.instanceCustomIndex];
    GpuOpenPbrMaterial mat  = materials[mesh.materialId];

    float3 V = -ray.direction;
    // Face-forwarded geometric normal (used for ray origin offset to prevent self-intersection)
    float3 geoNormal = (dot(hit.normal, V) < 0.0f) ? -hit.normal : hit.normal;
    float3 normal = geoNormal;

    // Tangent-space normal map override
    if (mat.normalTexId != 0xFFFFFFFFu)
    {
        float3 nTex = g_Textures[NonUniformResourceIndex(mat.normalTexId)].SampleLevel(hit.uv, 0.0f).xyz;
        float3 localN = normalize(nTex * 2.0f - 1.0f);
        normal = normalize(toWorld(localN, geoNormal));

        // Clamp shading normal if normal map tilts it behind the view vector V
        if (dot(normal, V) < 1e-4f)
        {
            normal = geoNormal;
        }
    }

    float3 baseRgb = mat.baseColorCoeffs.rgb;
    if (mat.baseColorTexId != 0xFFFFFFFFu)
    {
        baseRgb *= g_Textures[NonUniformResourceIndex(mat.baseColorTexId)].SampleLevel(hit.uv, 0.0f).rgb;
    }
    float4 baseReflectance = rgbToSpectrum4(baseRgb, ray.wavelengths);
    float4 specReflectance = rgbToSpectrum4(mat.specularColorCoeffs.rgb, ray.wavelengths);

    // Specular & Coat Roughness texture overrides
    if (mat.specularRoughnessTexId != 0xFFFFFFFFu)
    {
        mat.specularRoughness = g_Textures[NonUniformResourceIndex(mat.specularRoughnessTexId)].SampleLevel(hit.uv, 0.0f).r;
    }
    if (mat.coatRoughnessTexId != 0xFFFFFFFFu)
    {
        mat.coatRoughness = g_Textures[NonUniformResourceIndex(mat.coatRoughnessTexId)].SampleLevel(hit.uv, 0.0f).r;
    }

    // 3. Emission Lobe
    if (mat.emissionAndLuminance.w > 0.0f && any(mat.emissionAndLuminance.xyz > 0.0f))
    {
        float4 emissionSpectrum = rgbToSpectrum4(mat.emissionAndLuminance.xyz, ray.wavelengths) * mat.emissionAndLuminance.w;
        float4 emittedRadiance  = ray.throughput * emissionSpectrum;
        float3 emittedRgb       = spectral4ToLinearRgb(emittedRadiance, ray.wavelengths);

        g_AccumulationImage[pixelCoord] += float4(emittedRgb, 0.0f);

        // Terminate pure non-scattering emissive surfaces
        if (mat.baseWeight <= 1e-4f && mat.specularWeight <= 1e-4f)
        {
            terminateRay(rayIndex, ray);
            return;
        }
    }

    // 4. OpenPBR Multi-Lobe Energy Partitioning
    float cosThetaV = clamp(dot(normal, V), 0.0f, 1.0f);

    float4 F_dielectric      = evalFresnelDielectric(cosThetaV, max(1.001f, mat.specularIor), float4(1.0f));
    float4 F_spec_dielectric = F_dielectric * mat.specularWeight;
    float4 F0_metal          = baseReflectance * specReflectance;
    float  schlickFactor     = pow(max(0.0f, 1.0f - cosThetaV), 5.0f);
    float4 F_metal           = F0_metal + (float4(1.0f) - F0_metal) * schlickFactor;
    float4 F                 = lerp(F_spec_dielectric, F_metal, mat.baseMetalness);

    float4 F_coat = float4(0.0f);
    float  pCoat  = 0.0f;
    if (mat.coatWeight > 0.0f)
    {
        F_coat = evalFresnelDielectric(cosThetaV, max(1.001f, mat.coatIor), float4(1.0f)) * mat.coatWeight;
        pCoat  = clamp(0.25f * dot(F_coat, float4(1.0f)), 0.0f, 1.0f);
    }
    float4 coatTransmission = max(float4(0.0f), float4(1.0f) - F_coat);

    float specEnergy = clamp(0.25f * dot(F, float4(1.0f)), 0.0f, 1.0f);
    float diffEnergy = mat.baseWeight * (1.0f - mat.baseMetalness)
                     * clamp(1.0f - 0.25f * dot(F_spec_dielectric, float4(1.0f)), 0.0f, 1.0f);
    float baseTotalEnergy = specEnergy + diffEnergy;

    if (pCoat < 1e-5f && baseTotalEnergy < 1e-5f)
    {
        terminateRay(rayIndex, ray);
        return;
    }

    // 5. Lobe Importance Sampling
    float3 nextDir;
    float4 throughputUpdate;
    float  xi = randomFloat(seed);

    if (xi < pCoat)
    {
        // 5a. Clearcoat Lobe (Specular Reflection)
        float coatRoughness = clamp(mat.coatRoughness, 0.001f, 1.0f);
        float alpha         = coatRoughness * coatRoughness;

        if (alpha < 0.002f)
        {
            nextDir          = reflect(-V, normal);
            throughputUpdate = F_coat / max(1e-4f, pCoat);
        }
        else
        {
            float VdotH, weight;
            if (!sampleGGXMicrofacet(randomFloat2(seed), alpha, V, normal, cosThetaV, nextDir, VdotH, weight))
            {
                terminateRay(rayIndex, ray);
                return;
            }

            float4 F_coat_H  = evalFresnelDielectric(VdotH, max(1.001f, mat.coatIor), float4(1.0f)) * mat.coatWeight;
            throughputUpdate = (F_coat_H * weight) / max(1e-4f, pCoat);
        }
    }
    else
    {
        float xiBase     = (xi - pCoat) / max(1e-4f, 1.0f - pCoat);
        float pSpec      = (mat.baseMetalness >= 0.99f)
                         ? 1.0f
                         : ((specEnergy <= 1e-5f) ? 0.0f : clamp(specEnergy / baseTotalEnergy, 0.05f, 0.95f));
        float pDiff      = 1.0f - pSpec;
        float invBasePdf = 1.0f / max(1e-4f, 1.0f - pCoat);

        if (xiBase < pSpec)
        {
            // 5b. Base Specular Lobe (GGX / Mirror)
            float roughness = clamp(mat.specularRoughness, 0.001f, 1.0f);
            float alpha     = roughness * roughness;

            if (alpha < 0.002f)
            {
                nextDir          = reflect(-V, normal);
                throughputUpdate = (F / pSpec) * coatTransmission * invBasePdf;
            }
            else
            {
                float VdotH, weight;
                if (!sampleGGXMicrofacet(randomFloat2(seed), alpha, V, normal, cosThetaV, nextDir, VdotH, weight))
                {
                    terminateRay(rayIndex, ray);
                    return;
                }

                float4 F_dielectric_H = evalFresnelDielectric(VdotH, max(1.001f, mat.specularIor), float4(1.0f));
                float  schlick_H      = pow(max(0.0f, 1.0f - VdotH), 5.0f);
                float4 F_metal_H      = F0_metal + (float4(1.0f) - F0_metal) * schlick_H;
                float4 F_H            = lerp(F_dielectric_H * mat.specularWeight, F_metal_H, mat.baseMetalness);

                throughputUpdate = (F_H * weight / pSpec) * coatTransmission * invBasePdf;
            }
        }
        else
        {
            // 5c. Base Diffuse Lobe (Lambertian)
            nextDir = sampleCosineHemisphere(randomFloat2(seed), normal);

            float4 diffuseReflectance = baseReflectance * mat.baseWeight * (1.0f - mat.baseMetalness)
                                      * max(float4(0.0f), float4(1.0f) - F_spec_dielectric);
            throughputUpdate = (diffuseReflectance / pDiff) * coatTransmission * invBasePdf;
        }
    }

    ray.throughput *= throughputUpdate;

    // 6. Russian Roulette Termination
    if (params.currentBounce >= 3)
    {
        float maxThroughput = max(ray.throughput.x, max(ray.throughput.y, max(ray.throughput.z, ray.throughput.w)));
        float q = clamp(maxThroughput, 0.05f, 0.95f);
        if (randomFloat(seed) > q)
        {
            terminateRay(rayIndex, ray);
            return;
        }
        ray.throughput /= q;
    }

    // 7. Next Bounce State Advancement
    ray.origin = hit.worldPos + geoNormal * 1e-3f; // Offset along true geometric normal
    ray.direction = nextDir;
    ray.tMin = 1e-3f;
    ray.tMax = 10000.0f;
    ray.bounceCount += 1;

    if (ray.bounceCount >= params.maxBounces)
    {
        ray.isTerminated = 1;
    }

    g_RayQueue[rayIndex] = ray;
}
