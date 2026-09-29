// Tonemap.slang
struct SceneParams
{
    float4x4 invView;
    float4x4 invProj;

    uint64_t vertexBufferAddress;
    uint64_t indexBufferAddress;

    float2 shiftOffset;
    float2 tiltAxis;

    float tiltAngle;
    float focalDistance;
    float apertureRadius;
    uint width;

    uint height;
    uint frameCounter;
    uint currentBounce;
    uint maxBounces;

    uint totalPixels;
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::push_constant]]
SceneParams params;

// Binding 1: Offscreen HDR Accumulation Image (RW Storage Image)
[[vk::binding(1, 0)]]
RWTexture2D<float4> accumulationImage;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= params.width || id.y >= params.height)
        return;

    int2 pixelCoord = int2(id.xy);

    // Read stored HDR / accumulated spectral radiance
    float4 rawColor = accumulationImage[pixelCoord];

    // --- TODO: Implement Spectral/XYZ -> sRGB conversion, tonemapping curve, gamma correction ---
    float4 tonemappedColor = rawColor;

    // Write back in-place before the swapchain blit pass
    accumulationImage[pixelCoord] = tonemappedColor;
}