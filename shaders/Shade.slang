import Common;

[[vk::push_constant]]
ConstantBuffer<SceneParams> params;

// Set 0 Bindings
[[vk::binding(1, 0)]]
RWTexture2D<float4> g_AccumulationImage;

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
    HitPayload hit = g_HitQueue[rayIndex];

    if (ray.isTerminated != 0)
        return;

    uint2 pixelCoord = GetPixelCoord(ray.pixelIndex, params.width);

    // ==========================================
    // CASE A: RAY MISSED (Sky / Environment)
    // ==========================================
    if (hit.hitStatus == 0)
    {
        // ---------------- PSEUDOCODE: Spectral Environment ----------------
        // 1. Evaluate sky radiance at the 4 sampled wavelengths:
        //    float4 skyRadiance = evaluateSpectralSky(ray.direction, ray.wavelengths);
        // 2. Deposit spectral energy:
        //    float4 depositedLight = skyRadiance * ray.throughput;
        // 3. Convert 4 spectral bins to CIE XYZ using color matching functions (CMF):
        //    float3 xyz = spectralToXYZ(depositedLight, ray.wavelengths);
        // ------------------------------------------------------------------
        
        float4 accumulatedColor = g_AccumulationImage[pixelCoord];
        accumulatedColor += float4(0.2, 0.4, 0.8, 1.0); // Pseudocode: sky deposit
        g_AccumulationImage[pixelCoord] = accumulatedColor;

        ray.isTerminated = 1;
        g_RayQueue[rayIndex] = ray;
        return;
    }

    // ==========================================
    // CASE B: RAY HIT GEOMETRY (BSDF Evaluation)
    // ==========================================
    
    // ---------------- PSEUDOCODE: Material & Next Bounce ----------------
    // 1. Fetch material at hit.instanceCustomIndex
    // 2. If material is emissive:
    //      deposit light * ray.throughput to g_AccumulationImage;
    //      ray.isTerminated = 1;
    // 3. Sample Spectral BSDF (e.g. glass with dispersion, diffuse, conductor):
    //      float3 nextDir;
    //      float4 bsdfWeight;
    //      sampleBSDF(hit.normal, -ray.direction, ray.wavelengths, nextDir, bsdfWeight);
    // 4. Russian Roulette termination based on throughput luminance
    // --------------------------------------------------------------------

    // Prepare next bounce ray state:
    ray.origin      = hit.worldPos + hit.normal * 0.001f; // Offset along normal
    ray.direction   = reflect(ray.direction, hit.normal); // Pseudocode direction
    ray.throughput *= float4(0.8, 0.8, 0.8, 0.8);        // Dim throughput
    ray.bounceCount++;

    if (ray.bounceCount >= params.maxBounces)
    {
        ray.isTerminated = 1;
    }

    // Write back to queue for the next loop iteration in C++
    g_RayQueue[rayIndex] = ray;
}