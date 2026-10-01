// Common.slang
module Common;

public struct RayPayload
{
    public float3 origin;
    public float tMin;
    public float3 direction;
    public float tMax;

    public float4 wavelengths;
    public float4 throughput;

    public uint pixelIndex;
    public uint bounceCount;
    public uint isTerminated;
    public uint padding;
};

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

public struct SceneParams
{
    public float4x4 invView;
    public float4x4 invProj;

    public uint64_t vertexBufferAddress;
    public uint64_t indexBufferAddress;

    public float2 shiftOffset;
    public float2 tiltAxis;

    public float tiltAngle;
    public float focalDistance;
    public float apertureRadius;
    public uint width;

    public uint height;
    public uint frameCounter;
    public uint currentBounce;
    public uint maxBounces;

    public uint totalPixels;
    public uint pad0;
    public uint pad1;
    public uint pad2;
};

public struct GpuMeshDesc
{
    public uint64_t positionBufferAddress;
    public uint64_t attributtesBufferAddress; // Matches C++ header spelling
    public uint64_t indexBufferAddress;
    public uint materialId;
    public uint padding;
};

public struct VertexAttributes
{
    public float3 norm;
    public float2 uv;
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