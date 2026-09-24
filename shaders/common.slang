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

    /* ---------- Camera Options ---------- */
    public float apertureRadius; // 0.0 = sharp pinhole
    public float focalDistance;  // Distance to in-focus plane
    public float2 shiftOffset;   // Lens shift in NDC units 
    public float tiltAngle;      // Tilt angle in radians 
    public float2 tiltAxis;      // Normalized 2D rotation axis
};

// Ray State Queue Item
public struct RayPayload
{
    public float3 origin;
    public float tMin;
    public float3 direction;
    public float tMax;
    
    public float4 wavelengths;
    public float4 throughput; // Spectral power distribution (SPD) weight per bin
    
    public uint pixelIndex;
    public uint bounceCount;
    public uint isTerminated; 
    public uint padding;
};

// Intersection State Queue Item
public struct HitPayload
{
    public float3 worldPos;
    public uint instanceCustomIndex;
    public float3 normal;
    public uint primitiveId;
    public float2 uv;
    public uint hitStatus; 
    public uint padding;
};

// Convert 1D thread index to 2D pixel coordinate
public uint2 GetPixelCoord(uint pixelIndex, uint width)
{
    return uint2(pixelIndex % width, pixelIndex / width);
}

uint pcg_hash(inout uint state)
{
    uint prev = state;
    state = prev * 747796405u + 2891336453u;
    uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    return (word >> 22u) ^ word;
}

public float randomFloat(inout uint seed)
{ 
    uint r = pcg_hash(seed);
    // Multiply by 1.0 / 2^32 to map cleanly to a [0, 1) float
    return float(r) * (1.0 / 4294967296.0);
}

public float2 randomFloat2(inout uint seed)
{
    return { randomFloat(seed), randomFloat(seed) };
};

public float3 randomFloat3(inout uint seed)
{
    return { randomFloat(seed), randomFloat(seed), randomFloat(seed) };
};
public float4 randomFloat4(inout uint seed)
{
    return { randomFloat(seed), randomFloat(seed), randomFloat(seed), randomFloat(seed) };
};

// Uniformly maps a unit square sample [0, 1]^2 to a concentric unit disk [-1, 1]^2
public float2 sampleConcentricDisk(float2 u)
{
    float2 uOffset = 2.0f * u - float2(1.0f, 1.0f);
    if (uOffset.x == 0.0f && uOffset.y == 0.0f)
        return float2(0.0f, 0.0f);

    float theta, r;
    if (abs(uOffset.x) > abs(uOffset.y))
    {
        r = uOffset.x;
        theta = (3.14159265f / 4.0f) * (uOffset.y / uOffset.x);
    }
    else
    {
        r = uOffset.y;
        theta = (3.14159265f / 2.0f) - (3.14159265f / 4.0f) * (uOffset.x / uOffset.y);
    }
    return r * float2(cos(theta), sin(theta));
}