import Common;

[[vk::push_constant]]
ConstantBuffer<SceneParams> params;

// Buffer bindings matching Descriptor Set 0
[[vk::binding(2, 0)]]
RWStructuredBuffer<RayPayload> g_RayQueue;

float3 generatePerspectiveRay(
    float2 uv,
    float2 lensSample,
    out float3 rayOrigin,
    out float3 rayDir)
{
    // normalized device coordinates
    float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f) + params.shiftOffset;

    float4 target = mul(params.invProj, float4(ndc, 1.0f, 1.0f));
    float3 pinholeDirView = normalize(target.xyz / target.w);

    /* ------ Focus Target Point ------ */
    float3 focusPointView;
    if (params.tiltAngle != 0.0f)
    {
        // Rotate the focal plane normal according to tiltAxis and tiltAngle
        float sinT = sin(params.tiltAngle);
        float cosT = cos(params.tiltAngle);
        float2 axis = normalize(params.tiltAxis);
        float3 tiltedNormal = normalize(float3(axis.x * sinT, axis.y * sinT, -cosT));

        // Intersect pinhole ray with the tilted focal plane: dot(P - P0, N) = 0
        float3 p0 = float3(0.0f, 0.0f, -params.focalDistance);
        float t = dot(p0, tiltedNormal) / dot(pinholeDirView, tiltedNormal);
        focusPointView = pinholeDirView * t;
    }
    else
    {   
        // Standard parallel focus plane 
        focusPointView = pinholeDirView * (params.focalDistance / abs(pinholeDirView.z));
    }

    /* ------ Aperture Disk ------ */
    float3 lensOffsetView = float3(0.0f);
    if (params.apertureRadius > 0.0f)
    {
        float2 lensDisk = sampleConcentricDisk(lensSample) * params.apertureRadius;
        lensOffsetView = float3(lensDisk.x, lensDisk.y, 0.0f);
    }

    // Direction from aperture sample location to focus point
    float3 rayDirView = normalize(focusPointView - lensOffsetView);

    // Transform lens origin and ray direction to World Space
    rayOrigin = mul(params.invView, float4(lensOffsetView, 1.0f)).xyz;
    rayDir = mul(params.invView, float4(rayDirView, 0.0f)).xyz;
}

float4 generateWavelengths(float hero, float2 spectrum)
{
    float span = spectrum.y - spectrum.x;
    float step = (span / 4);

    float4 wavelengths;
    wavelengths.x = fmod(hero - spectrum.x + step, span) + spectrum.x;
    wavelengths.y = fmod(hero - spectrum.x + 2.0f * step, span) + spectrum.x;
    wavelengths.z = fmod(hero - spectrum.x + 3.0f * step, span) + spectrum.x;
    wavelengths.w = fmod(hero - spectrum.x + 4.0f * step, span) + spectrum.x;
    return wavelengths;
}


[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint rayIndex = dispatchThreadID.x;
    if (rayIndex >= params.totalPixels)
        return;

    uint seed = rayIndex + (params.frameCounter * params.totalPixels);

    

    uint2 pixelCoord = GetPixelCoord(rayIndex, params.width);
    float2 subpixelJitter = randomFloat2(seed);
    float2 lensSample = randomFloat2(seed);

    float2 uv = (float2(pixelCoord) + subpixelJitter) / float2(params.width, params.height);

    float3 rayOrigin, rayDir;
    generatePerspectiveRay(uv, lensSample, rayOrigin, rayDir);

    float2 visibleSpectrum = { 380.0f, 780.0f };
    float heroWavelength = lerp(visibleSpectrum.x, visibleSpectrum.y, randomFloat(seed));

    RayPayload ray;
    ray.origin = rayOrigin;
    ray.tMin = 0.001f;
    ray.direction = rayDir;
    ray.tMax = 10000.0f;
    ray.wavelengths = generateWavelengths(heroWavelength, visibleSpectrum);;
    ray.throughput = float4(1.0f, 1.0f, 1.0f, 1.0f);
    ray.pixelIndex = rayIndex;
    ray.bounceCount = 0;
    ray.isTerminated = 0;
    ray.padding = 0;

    // Push into the Ray Queue
    g_RayQueue[rayIndex] = ray;
}