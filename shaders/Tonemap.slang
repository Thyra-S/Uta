import Common;

[[vk::push_constant]]
ConstantBuffer<SceneParams> params;

[[vk::binding(1, 0)]] RWTexture2D<float4> g_AccumulationImage;
[[vk::binding(4, 0)]] RWTexture2D<float4> g_ResolvedImage;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= params.width || id.y >= params.height)
        return;

    int2 pixelCoord = int2(id.xy);

    // 1. Fetch raw accumulated sum
    float3 sumRadiance = g_AccumulationImage[pixelCoord].rgb;

    // 2. Compute sample mean
    float totalSamples = max(1.0, float(params.frameCounter + 1));
    float3 linearColor = sumRadiance / totalSamples;

    // 3. Tonemapping (Reinhard)
    float3 mapped = linearColor / (linearColor + float3(1.0, 1.0, 1.0));

    // 4. Gamma Correction (Linear -> sRGB approximation)
    float3 srgb = pow(max(mapped, float3(0.0, 0.0, 0.0)), float3(1.0 / 2.2));

    g_ResolvedImage[pixelCoord] = float4(srgb, 1.0);
}