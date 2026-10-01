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

    // ==========================================
    // CASE A: RAY MISSED (Black)
    // ==========================================
    if (hit.hitStatus == 0)
    {
        g_AccumulationImage[pixelCoord] = float4(0.0, 0.0, 0.0, 1.0);
        ray.isTerminated = 1;
        g_RayQueue[rayIndex] = ray;
        return;
    }

    // ==========================================
    // CASE B: RAY HIT GEOMETRY (Normal / Solid Color)
    // ==========================================

    // Option 1: Interpolate and visualize smooth surface normal
    GpuMeshDesc* meshDescs = (GpuMeshDesc*)params.vertexBufferAddress;
    GpuMeshDesc mesh = meshDescs[hit.instanceCustomIndex];

    uint3* indices = (uint3*)mesh.indexBufferAddress;
    uint3 tri = indices[hit.primitiveId];

    VertexAttributes* attrs = (VertexAttributes*)mesh.attributtesBufferAddress;
    VertexAttributes a0 = attrs[tri.x];
    VertexAttributes a1 = attrs[tri.y];
    VertexAttributes a2 = attrs[tri.z];

    // Barycentric interpolation: u = hit.uv.x, v = hit.uv.y
    float b0 = 1.0 - hit.uv.x - hit.uv.y;
    float b1 = hit.uv.x;
    float b2 = hit.uv.y;

    float3 normal = normalize(b0 * a0.norm + b1 * a1.norm + b2 * a2.norm);

    // Map normal [-1, 1] -> [0, 1] RGB
    float3 displayColor = normal * 0.5 + 0.5;

    float3 newSampleRadiance = displayColor;

    // Accumulate raw sum of radiance
    if (params.frameCounter == 0)
    {
        g_AccumulationImage[pixelCoord] = float4(newSampleRadiance, 1.0);
    }
    else
    {
        g_AccumulationImage[pixelCoord] += float4(newSampleRadiance, 0.0);
    }

    ray.isTerminated = 1;
    g_RayQueue[rayIndex] = ray;
}