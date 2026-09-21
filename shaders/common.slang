// Common.slang
module Common;

// Push Constants
public struct SceneParams
{
    public float4x4 invView;
    public float4x4 invProj;
    public uint frameCounter;
    public uint currentBounce;
    public uint maxBounces;
    public uint totalPixels;
    public uint width;
    public uint height;
};

// 1. Ray State Queue Item
public struct RayPayload
{
    public float3 origin;
    public float tMin;
    public float3 direction;
    public float tMax;
    
    // Spectral bins: [0] = Hero wavelength in nm, [1..3] = 3 shifted companion wavelengths
    public float4 wavelengths;
    public float4 throughput; // Spectral power distribution (SPD) weight per bin
    
    public uint pixelIndex;
    public uint bounceCount;
    public uint isTerminated; // 1 = dead ray, 0 = active
    public uint padding;
};

// 2. Intersection State Queue Item
public struct HitPayload
{
    public float3 worldPos;
    public uint instanceCustomIndex;
    public float3 normal;
    public uint primitiveId;
    public float2 uv;
    public uint hitStatus; // 0 = Miss (Sky/Environment), 1 = Triangle Hit
    public uint padding;
};

// Convert 1D thread index to 2D pixel coordinate
public uint2 GetPixelCoord(uint pixelIndex, uint width)
{
    return uint2(pixelIndex % width, pixelIndex / width);
}