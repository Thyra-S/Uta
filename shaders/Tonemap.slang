// Tonemap.slang
import Common;

[[vk::push_constant]]
ConstantBuffer<SceneParams> params;

// Descriptor Set 0 Bindings
[[vk::binding(1, 0)]] RWTexture2D<float4> g_AccumulationImage;
[[vk::binding(4, 0)]] RWTexture2D<float4> g_ResolvedImage;

// ACES Film Tone Reproduction (Narkowicz fit)
float3 acesFilm(float3 x)
{
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

// IEC 61966-2-1 standard piecewise linear-to-sRGB transfer function
float3 linearToSrgb(float3 c)
{
    float3 lo = 12.92f * c;
    float3 hi = 1.055f * pow(max(c, float3(0.0f)), 1.0f / 2.4f) - 0.055f;
    return select(c <= 0.0031308f, lo, hi);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= params.width || id.y >= params.height)
        return;

    int2 pixelCoord = int2(id.xy);

    // 1. Compute Monte Carlo sample mean of accumulated spectral-to-sRGB radiance
    float3 sumRadiance  = g_AccumulationImage[pixelCoord].rgb;
    float  totalSamples = max(1.0f, float(params.frameCounter + 1));
    float3 linearColor  = sumRadiance / totalSamples;

    // 2. Exposure compensation (reads host exposure from params.pad1, defaults to 1.0)
    float  hostExposure = asfloat(params.pad1);
    float  exposure     = (hostExposure > 0.0f) ? hostExposure : 1.0f;
    float3 exposedColor = max(float3(0.0f), linearColor * exposure);

    // 3. Output encoding (scRGB HDR linear vs. ACES + sRGB SDR)
    bool   isHdrScRgb  = (params.pad0 == 1);
    float3 outputColor = isHdrScRgb ? exposedColor : linearToSrgb(acesFilm(exposedColor));

    g_ResolvedImage[pixelCoord] = float4(outputColor, 1.0f);
}
