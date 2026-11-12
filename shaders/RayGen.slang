// RayGen.slang
import Common;

[[vk::push_constant]]
ConstantBuffer<SceneParams> params;

// Descriptor Set 0 Bindings
[[vk::binding(2, 0)]] RWStructuredBuffer<RayPayload> g_RayQueue;

void generatePerspectiveRay(
    float2     uv,
    float2     lensSample,
    out float3 rayOrigin,
    out float3 rayDir)
{
    // Normalized device coordinates with lens shift
    float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f) + params.shiftOffset;

    float4 target = mul(params.invProj, float4(ndc, 1.0f, 1.0f));
    float3 pinholeDirView = normalize(target.xyz / target.w);

    // 1. Focus Target Point (Scheimpflug tilt or standard parallel plane)
    float3 focusPointView;
    if (params.tiltAngle != 0.0f)
    {
        float sinT = sin(params.tiltAngle);
        float cosT = cos(params.tiltAngle);
        float2 axis = normalize(params.tiltAxis);
        float3 tiltedNormal = normalize(float3(axis.x * sinT, axis.y * sinT, -cosT));

        // Intersect pinhole ray with tilted focal plane: dot(P - P0, N) = 0
        float3 p0 = float3(0.0f, 0.0f, -params.focalDistance);
        float t = dot(p0, tiltedNormal) / dot(pinholeDirView, tiltedNormal);
        focusPointView = pinholeDirView * t;
    }
    else
    {
        focusPointView = pinholeDirView * (params.focalDistance / abs(pinholeDirView.z));
    }

    // 2. Aperture Disk Offset
    float3 lensOffsetView = float3(0.0f);
    if (params.apertureRadius > 0.0f)
    {
        float2 lensDisk = sampleConcentricDisk(lensSample) * params.apertureRadius;
        lensOffsetView = float3(lensDisk.x, lensDisk.y, 0.0f);
    }

    // 3. Transform lens origin and ray direction to World Space
    float3 rayDirView = normalize(focusPointView - lensOffsetView);
    rayOrigin = mul(params.invView, float4(lensOffsetView, 1.0f)).xyz;
    rayDir    = mul(params.invView, float4(rayDirView, 0.0f)).xyz;
}

float4 generateWavelengths(float hero, float2 spectrum)
{
    float span = spectrum.y - spectrum.x;
    float step = span * 0.25f;

    return float4(
        hero,
        fmod(hero - spectrum.x + 1.0f * step, span) + spectrum.x,
        fmod(hero - spectrum.x + 2.0f * step, span) + spectrum.x,
        fmod(hero - spectrum.x + 3.0f * step, span) + spectrum.x
    );
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint2 pixelCoord = dispatchThreadID.xy;
    if (pixelCoord.x >= params.width || pixelCoord.y >= params.height)
        return;

    uint rayIndex = pixelCoord.y * params.width + pixelCoord.x;
    uint seed = rayIndex + (params.frameCounter * params.totalPixels);

    float2 subpixelJitter = randomFloat2(seed);
    float2 lensSample     = randomFloat2(seed);
    float2 uv             = (float2(pixelCoord) + subpixelJitter) / float2(params.width, params.height);

    float3 rayOrigin;
    float3 rayDir;
    generatePerspectiveRay(uv, lensSample, rayOrigin, rayDir);

    float2 visibleSpectrum = float2(380.0f, 780.0f);
    float heroWavelength = lerp(visibleSpectrum.x, visibleSpectrum.y, randomFloat(seed));

    RayPayload ray;
    ray.origin       = rayOrigin;
    ray.tMin         = 0.001f;
    ray.direction    = rayDir;
    ray.tMax         = 10000.0f;
    ray.wavelengths  = generateWavelengths(heroWavelength, visibleSpectrum);
    ray.throughput   = float4(1.0f);
    ray.pixelIndex   = rayIndex;
    ray.bounceCount  = 0;
    ray.isTerminated = 0;
    ray.padding      = 0;

    g_RayQueue[rayIndex] = ray;
}
