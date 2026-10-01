import Common;

[[vk::push_constant]]
ConstantBuffer<SceneParams> params;

// Set 0 Bindings
[[vk::binding(0, 0)]] RaytracingAccelerationStructure g_TLAS;
[[vk::binding(2, 0)]] RWStructuredBuffer<RayPayload>  g_RayQueue;
[[vk::binding(3, 0)]] RWStructuredBuffer<HitPayload>  g_HitQueue;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint rayIndex = dispatchThreadID.x;
    if (rayIndex >= params.totalPixels)
        return;

    RayPayload ray = g_RayQueue[rayIndex];

    // Zero-initialize hit payload (hitStatus = 0 defaults to Miss/Environment)
    HitPayload hit = (HitPayload)0;

    // Skip inactive or terminated rays
    if (ray.isTerminated != 0)
    {
        g_HitQueue[rayIndex] = hit;
        return;
    }

    // 1. Setup ray descriptor
    RayDesc rayDesc;
    rayDesc.Origin    = ray.origin;
    rayDesc.Direction = ray.direction;
    rayDesc.TMin      = ray.tMin;
    rayDesc.TMax      = ray.tMax;

    // 2. Hardware inline ray query traversal
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(g_TLAS, RAY_FLAG_FORCE_OPAQUE, 0xFF, rayDesc);

    while (query.Proceed())
    {
        // Traversal handled by hardware RT cores
    }

    // 3. Populate Hit Queue
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        hit.hitStatus             = 1;
        // CommittedInstanceIndex() gives 0-based instance index matching meshDescBuffer
        hit.instanceCustomIndex   = query.CommittedInstanceIndex();
        hit.primitiveId           = query.CommittedPrimitiveIndex();
        hit.uv                    = query.CommittedTriangleBarycentrics(); // float2 (u, v)
        hit.worldPos              = ray.origin + ray.direction * query.CommittedRayT();
        hit.normal                = float3(0.0, 0.0, 0.0); // Resolved in Shade.slang
    }

    g_HitQueue[rayIndex] = hit;
}