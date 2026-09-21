import Common;

[[vk::push_constant]]
ConstantBuffer<SceneParams> params;

// Buffer bindings matching Descriptor Set 0
[[vk::binding(2, 0)]]
RWStructuredBuffer<RayPayload> g_RayQueue;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint rayIndex = dispatchThreadID.x;
    if (rayIndex >= params.totalPixels)
        return;

    uint2 pixelCoord = GetPixelCoord(rayIndex, params.width);

    // ---------------- PSEUDOCODE: Camera & Spectral Init ----------------
    // 1. Compute jittered subpixel ray for anti-aliasing
    //    float2 uv = (float2(pixelCoord) + randomSubpixelOffset()) / float2(params.width, params.height);
    //    float3 rayOrigin = mul(params.invView, float4(0, 0, 0, 1)).xyz;
    //    float3 rayDir    = generatePerspectiveRay(uv, params.invView, params.invProj);
    //
    // 2. Sample Hero Wavelength across visible spectrum (380nm - 780nm)
    //    float heroWavelength = lerp(380.0, 780.0, randomFloat());
    //    float4 wavelengths   = computeHeroWavelengthOffsets(heroWavelength);
    // --------------------------------------------------------------------

    RayPayload ray;
    ray.origin       = float3(0.0, 0.0, -2.0); // Pseudocode fallback
    ray.tMin         = 0.001f;
    ray.direction    = float3(0.0, 0.0, 1.0);  // Pseudocode fallback
    ray.tMax         = 10000.0f;
    ray.wavelengths  = float4(550.0, 470.0, 610.0, 680.0); // Sampled spectrum
    ray.throughput   = float4(1.0, 1.0, 1.0, 1.0);         // Initial energy = 100%
    ray.pixelIndex   = rayIndex;
    ray.bounceCount  = 0;
    ray.isTerminated = 0;
    ray.padding      = 0;

    // Push into the Ray Queue
    g_RayQueue[rayIndex] = ray;
}