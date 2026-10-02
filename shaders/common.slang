// Common.slang
module Common;

public struct GpuOpenPbrMaterial
{
    // BASE LOBE
    public float4 baseColorCoeffs;
    public float baseWeight;
    public float baseRoughness;
    public float baseMetalness;
    public float baseDiffuseRoughness;

    // SPECULAR LOBE
    public float4 specularColorCoeffs;
    public float specularWeight;
    public float specularRoughness;
    public float specularIor;
    public float specularRoughnessAnisotropy;

    // TRANSMISSION LOBE
    public float4 transmissionColorCoeffs;
    public float4 transmissionScatterAndDispersion;
    public float transmissionWeight;
    public float transmissionDepth;
    public float padTrans0;
    public float padTrans1;

    // SUBSURFACE LOBE
    public float4 subsurfaceColorAndWeight;
    public float4 subsurfaceRadiusAndScale;

    // COAT LOBE
    public float4 coatColorAndDarkening;
    public float coatWeight;
    public float coatRoughness;
    public float coatIor;
    public float padCoat;

    // EMISSION & GEOMETRY
    public float4 emissionAndLuminance;
    public uint baseColorTexId;
    public uint specularRoughnessTexId;
    public uint coatRoughnessTexId;
    public uint normalTexId;

    public uint thinWalled;
    public uint pad0;
    public uint pad1;
    public uint pad2;
};

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
    public float pad0;
    public float2 uv;
    public float2 pad1;
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

public void buildOrthonormalBasis(float3 normal, out float3 tangent, out float3 bitangent)
{
    float3 up = abs(normal.z) < 0.999f ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
    tangent = normalize(cross(up, normal));
    bitangent = cross(normal, tangent);
}

public float3 toWorld(float3 localDir, float3 normal)
{
    float3 tangent, bitangent;
    buildOrthonormalBasis(normal, tangent, bitangent);
    return localDir.x * tangent + localDir.y * bitangent + localDir.z * normal;
}
public float3 sampleCosineHemisphere(float2 u, float3 normal)
{
    float2 diskSample = sampleConcentricDisk(u);
    float z = sqrt(max(0.0f, 1.0f - diskSample.x * diskSample.x - diskSample.y * diskSample.y));
    float3 localDir = normalize(float3(diskSample.x, diskSample.y, z));
    return toWorld(localDir, normal);
}

// Evaluates a polynomial spectral curve: rho(lambda) = c0*x^2 + c1*x + c2 scaled by scale
// lambda is in nm [380.0, 780.0]
public float evaluateSpectralPolynomial(float4 coeffs, float lambda)
{
    float x = (lambda - 380.0f) * (1.0f / 400.0f); // Normalize to [0, 1]
    float p = coeffs.x * x * x + coeffs.y * x + coeffs.z;
    // Sigmoidal clamp to enforce energy conservation [0, 1]
    float s = 0.5f + 0.5f * (p / sqrt(1.0f + p * p));
    return clamp(s * coeffs.w, 0.0f, 1.0f);
}

public float4 evaluateSpectrum4(float4 coeffs, float4 wavelengths)
{
    return float4(
        evaluateSpectralPolynomial(coeffs, wavelengths.x),
        evaluateSpectralPolynomial(coeffs, wavelengths.y),
                  evaluateSpectralPolynomial(coeffs, wavelengths.z),
        evaluateSpectralPolynomial(coeffs, wavelengths.w)
    );
}

// Dielectric Schlick Fresnel evaluated per wavelength
public float4 evalFresnelDielectric(float cosTheta, float ior, float4 tint)
{
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    float schlick = r0 + (1.0f - r0) * pow(max(0.0f, 1.0f - cosTheta), 5.0f);
    return tint * schlick;
}