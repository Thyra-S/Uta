import Common;

[[vk::push_constant]]
ConstantBuffer<SceneParams> params;

// Set 0 Bindings
[[vk::binding(1, 0)]] RWTexture2D<float4>            g_AccumulationImage;
[[vk::binding(2, 0)]] RWStructuredBuffer<RayPayload> g_RayQueue;
[[vk::binding(3, 0)]] RWStructuredBuffer<HitPayload> g_HitQueue;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint rayIndex = dispatchThreadID.x;
    if (rayIndex >= params.totalPixels)
        return;

    RayPayload ray = g_RayQueue[rayIndex];
    HitPayload hit = g_HitQueue[rayIndex];

    if (ray.isTerminated != 0)
        return;

    uint2 pixelCoord = uint2(ray.pixelIndex % params.width, ray.pixelIndex / params.width);

    /* ---- RAY MISS ---- */
    if (hit.hitStatus == 0)
    {
        g_AccumulationImage[pixelCoord] = float4(0.0, 0.0, 0.0, 1.0);
        ray.isTerminated = 1;
        g_RayQueue[rayIndex] = ray;
        return;
    }

    /* ---- RAY HIT ---- */
    // Evaluate BRSDF and generate new ray direction

    // Option 1: Interpolate and visualize smooth surface normal
    GpuMeshDesc* meshDescs = (GpuMeshDesc*)params.vertexBufferAddress;
    GpuMeshDesc mesh = meshDescs[hit.instanceCustomIndex];

    uint *indices = (uint *)mesh.indexBufferAddress;
    uint3 tri = uint3
    (
    indices[hit.primitiveId * 3 + 0],
    indices[hit.primitiveId * 3 + 1],
    indices[hit.primitiveId * 3 + 2]
    );

    VertexAttributes* attrs = (VertexAttributes*)mesh.attributtesBufferAddress;
    VertexAttributes a0 = attrs[tri.x];
    VertexAttributes a1 = attrs[tri.y];
    VertexAttributes a2 = attrs[tri.z];

    hit.normal = normalize(a0.norm * (1.0f - hit.uv.x - hit.uv.y) + a1.norm * hit.uv.x + a2.norm * hit.uv.y);
   
    uint seed = rayIndex + (params.frameCounter * params.totalPixels + params.currentBounce * params.totalPixels);

    float pdf;
    float3 localDir = sampleCosineHemisphere(randomFloat2(seed), pdf);
    float3 nextDir = normalize(toWorld(localDir, hit.normal));

    // Spawn next bounce ray
    ray.origin = hit.worldPos + hit.normal * 0.001f; // offset to avoid self-intersection
    ray.direction = nextDir;
    ray.bounceCount += 1;
    
    // Spectral throughput:
    // For ideal Lambertian diffuse, (BRDF * cosTheta) / pdf cancels out to albedo.
    // If your surface albedo is 1.0 across all wavelengths:
    // ray.throughput remains unchanged.
    float3 rgbAlbedo, coeffs;
    rgbAlbedo = g_AlbedoTexture.Sample(uv);
    coeffs = Sample3DLUT();

    spectralAlbedo.x = rgbToSpectrum(rgbAlbedo, ray.wavelengths.x);
    spectralAlbedo.y = rgbToSpectrum(rgbAlbedo, ray.wavelengths.y);
    spectralAlbedo.z = rgbToSpectrum(rgbAlbedo, ray.wavelengths.z);
    spectralAlbedo.w = rgbToSpectrum(rgbAlbedo, ray.wavelengths.w);

    ray.throughput *= albedo;

    g_RayQueue[rayIndex] = ray;
}