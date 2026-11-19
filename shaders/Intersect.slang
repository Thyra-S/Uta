// Intersect.slang
import Common;

[[vk::push_constant]]
ConstantBuffer<SceneParams> params;

// Descriptor Set 0 Bindings
[[vk::binding(0, 0)]] RaytracingAccelerationStructure g_TLAS;
[[vk::binding(2, 0)]] RWStructuredBuffer<RayPayload>  g_RayQueue;
[[vk::binding(3, 0)]] RWStructuredBuffer<HitPayload>  g_HitQueue;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixelCoord = dispatchThreadID.xy;
    if (pixelCoord.x >= params.width || pixelCoord.y >= params.height)
        return;

    uint rayIndex = pixelCoord.y * params.width + pixelCoord.x;
    RayPayload ray = g_RayQueue[rayIndex];
    HitPayload hit = (HitPayload)0; // hitStatus = 0 defaults to Miss/Environment

    if (isRayTerminated(ray.bounceCount))
    {
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

    // 3. Populate Hit Queue on triangle intersection
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        hit.hitStatus           = 1;
        hit.instanceCustomIndex = query.CommittedInstanceIndex();
        hit.primitiveId         = query.CommittedPrimitiveIndex();
        hit.uv                  = query.CommittedTriangleBarycentrics();
        hit.worldPos            = ray.origin + ray.direction * query.CommittedRayT();

        float3x4 objToWorld = query.CommittedObjectToWorld3x4();
        float3x3 worldMat = float3x3(
            objToWorld[0].xyz,
            objToWorld[1].xyz,
            objToWorld[2].xyz
        );

        GpuMeshDesc* meshDescs = (GpuMeshDesc*)params.vertexBufferAddress;
        GpuMeshDesc  mesh      = meshDescs[hit.instanceCustomIndex];

        uint* indices = (uint*)mesh.indexBufferAddress;
        uint3 tri = uint3(
            indices[hit.primitiveId * 3 + 0],
            indices[hit.primitiveId * 3 + 1],
            indices[hit.primitiveId * 3 + 2]
        );
        float3 *positions = (float3 *)mesh.positionBufferAddress;
        float3 localGeomNormal = cross(
            positions[tri.y] - positions[tri.x],
            positions[tri.z] - positions[tri.x]
        );
        float3 worldGeomNormal = normalize(mul(worldMat, localGeomNormal));

        VertexAttributes *attrs = (VertexAttributes *)mesh.attributtesBufferAddress;
        float w0 = 1.0f - hit.uv.x - hit.uv.y;
        float3 localNormal = attrs[tri.x].norm * w0
                           + attrs[tri.y].norm * hit.uv.x
                           + attrs[tri.z].norm * hit.uv.y;

        if (dot(localNormal, localNormal) < 1e-6f)
        {
            localNormal = localGeomNormal;
        }

        float3 worldShadingNormal = normalize(mul(worldMat, localNormal));

        // Ensure smooth shading normal agrees with the true polygon winding
        if (dot(worldShadingNormal, worldGeomNormal) < 0.0f)
        {
            worldShadingNormal = -worldShadingNormal;
        }

        // True front-face test using polygon winding, stored in hit.padding
        bool isFrontFace = (dot(worldGeomNormal, -ray.direction) >= 0.0f);
        hit.normal = isFrontFace ? worldShadingNormal : -worldShadingNormal;
        hit.padding = isFrontFace ? 1u : 0u;
    }

    g_HitQueue[rayIndex] = hit;
}
