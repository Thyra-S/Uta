import Common;

[[vk::push_constant]]
ConstantBuffer<SceneParams> params;

// Set 0 Bindings
[[vk::binding(0, 0)]]
RaytracingAccelerationStructure g_TLAS;

[[vk::binding(2, 0)]]
RWStructuredBuffer<RayPayload> g_RayQueue;

[[vk::binding(3, 0)]]
RWStructuredBuffer<HitPayload> g_HitQueue;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint rayIndex = dispatchThreadID.x;
    if (rayIndex >= params.totalPixels)
        return;

    RayPayload ray = g_RayQueue[rayIndex];

    HitPayload hit;
    hit.hitStatus = 0; // Default: Miss

    // If ray was terminated on previous bounce, skip expensive traversal
    if (ray.isTerminated != 0)
    {
        g_HitQueue[rayIndex] = hit;
        return;
    }

    // 1. Pack into RayDesc
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
        // Traversal handled by RT cores
    }

    // 3. Populate Hit Queue based on result
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        hit.hitStatus             = 1; // Hit
        hit.instanceCustomIndex   = query.CommittedInstanceID();
        hit.primitiveId           = query.CommittedPrimitiveIndex();
        hit.uv                    = query.CommittedTriangleBarycentrics();
        
        // Compute hit position: Origin + Direction * HitDistance
        hit.worldPos              = ray.origin + ray.direction * query.CommittedRayT();

        // ---------------- PSEUDOCODE: Geometric Normal ----------------
        // In full engine: fetch vertices from storage buffer using primitiveId,
        // compute cross product of triangle edges for geometric normal
        hit.normal                = float3(0, 1, 0); 
        // -------------------------------------------------------------
    }

    g_HitQueue[rayIndex] = hit;
}