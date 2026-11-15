// common.slang
module Common;

// ============================================================================
// GPU Data Structures (Host-Device Shared Layouts)
// ============================================================================

public struct GpuOpenPbrMaterial
{
    // Base Lobe
    public float4 baseColor;
    public float  baseWeight;
    public float  baseRoughness;
    public float  baseMetalness;
    public float  baseDiffuseRoughness;

    // Specular Lobe
    public float4 specularColor;
    public float  specularWeight;
    public float  specularRoughness;
    public float  specularIor;
    public float  specularRoughnessAnisotropy;

    // Transmission Lobe
    public float4 transmissionColor;
    public float4 transmissionScatterAndDispersion;
    public float  transmissionWeight;
    public float  transmissionDepth;
    public float  padTrans0;
    public float  padTrans1;

    // Subsurface Lobe
    public float4 subsurfaceColorAndWeight;
    public float4 subsurfaceRadiusAndScale;

    // Coat Lobe
    public float4 coatColorAndDarkening;
    public float  coatWeight;
    public float  coatRoughness;
    public float  coatIor;
    public float  padCoat;

    // Emission & Geometry
    public float4 emissionAndLuminance;
    public uint   baseColorTexId;
    public uint   specularRoughnessTexId;
    public uint   coatRoughnessTexId;
    public uint   normalTexId;

    public uint   thinWalled;
    public uint   pad0;
    public uint   pad1;
    public uint   pad2;
};

public struct RayPayload
{
    public float3 origin;
    public float  tMin;
    public float3 direction;
    public float  tMax;

    public float4 wavelengths;
    public float4 throughput;

    public uint   pixelIndex;
    public uint   bounceCount;
    public uint   isTerminated;
    public uint   padding;
};

public struct HitPayload
{
    public float3 worldPos;
    public uint   instanceCustomIndex;
    public float3 normal;
    public uint   primitiveId;
    public float2 uv;
    public uint   hitStatus;
    public uint   padding;
};

public struct SceneParams
{
    public float4x4 invView;
    public float4x4 invProj;

    public uint64_t vertexBufferAddress;
    public uint64_t materialBufferAddress;

    public float2   shiftOffset;
    public float2   tiltAxis;

    public float    tiltAngle;
    public float    focalDistance;
    public float    apertureRadius;
    public uint     width;

    public uint     height;
    public uint     frameCounter;
    public uint     currentBounce;
    public uint     maxBounces;

    public uint     totalPixels;
    public uint     pad0;
    public uint     pad1;
    public uint     pad2;
};

public struct GpuMeshDesc
{
    public uint64_t positionBufferAddress;
    public uint64_t attributtesBufferAddress; // Matches C++ header spelling
    public uint64_t indexBufferAddress;
    public uint     materialId;
    public uint     padding;
};

public struct VertexAttributes
{
    public float3 norm;
    public float  pad0;
    public float2 uv;
    public float2 pad1;
};

// ============================================================================
// Coordinate & Random Number Utilities
// ============================================================================

// Converts a 1D thread/ray index to a 2D pixel coordinate
public uint2 GetPixelCoord(uint pixelIndex, uint width)
{
    return uint2(pixelIndex % width, pixelIndex / width);
}

public uint pcg_hash(inout uint state)
{
    uint prev = state;
    state = prev * 747796405u + 2891336453u;
    uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    return (word >> 22u) ^ word;
}

public float randomFloat(inout uint seed)
{
    // Multiply by 1.0 / 2^32 to map to [0, 1)
    return float(pcg_hash(seed)) * (1.0f / 4294967296.0f);
}

public float2 randomFloat2(inout uint seed)
{
    return float2(randomFloat(seed), randomFloat(seed));
}

// ============================================================================
// Sampling & Geometry Utilities
// ============================================================================

// Maps a unit square sample [0, 1]^2 to a concentric unit disk [-1, 1]^2
public float2 sampleConcentricDisk(float2 u)
{
    float2 uOffset = 2.0f * u - 1.0f;
    if (uOffset.x == 0.0f && uOffset.y == 0.0f)
        return float2(0.0f);

    float theta;
    float r;
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

// Transforms a tangent-space direction into world space aligned with normal
public float3 toWorld(float3 localDir, float3 normal)
{
    float3 up = abs(normal.z) < 0.999f ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);
    return localDir.x * tangent + localDir.y * bitangent + localDir.z * normal;
}

public float3 sampleCosineHemisphere(float2 u, float3 normal)
{
    float2 disk = sampleConcentricDisk(u);
    float z = sqrt(max(0.0f, 1.0f - dot(disk, disk)));
    return toWorld(float3(disk.x, disk.y, z), normal);
}

// ============================================================================
// Spectral & Color Math
// ============================================================================

// Evaluates a polynomial spectral curve across 4 hero wavelengths in [380, 780] nm:
// rho(lambda) = sigmoid(c0 * x^2 + c1 * x + c2) * scale
public float4 evaluateSpectrum4(float4 coeffs, float4 wavelengths)
{
    float4 x = (wavelengths - 380.0f) * (1.0f / 400.0f);
    float4 p = coeffs.x * x * x + coeffs.y * x + coeffs.z;
    float4 s = 0.5f + 0.5f * (p / sqrt(1.0f + p * p));
    return clamp(s * coeffs.w, 0.0f, 1.0f);
}

// Dielectric Schlick Fresnel evaluated per wavelength
public float4 evalFresnelDielectric(float cosTheta, float ior, float4 tint)
{
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 *= r0;
    float schlick = r0 + (1.0f - r0) * pow(max(0.0f, 1.0f - cosTheta), 5.0f);
    return tint * schlick;
}

// Exact unpolarized dielectric Fresnel (returns 1.0 on Total Internal Reflection)
public float evalExactDielectricFresnel(float cosThetaI, float ni, float nt)
{
    float eta = ni / nt;
    float sinThetaI2 = max(0.0f, 1.0f - cosThetaI * cosThetaI);
    float sinThetaT2 = eta * eta * sinThetaI2;

    // Total Internal Reflection (ray cannot escape glass into air)
    if (sinThetaT2 >= 1.0f)
        return 1.0f;

    float cosThetaT = sqrt(max(0.0f, 1.0f - sinThetaT2));
    float rParl = (nt * cosThetaI - ni * cosThetaT) / (nt * cosThetaI + ni * cosThetaT);
    float rPerp = (ni * cosThetaI - nt * cosThetaT) / (ni * cosThetaI + nt * cosThetaT);
    return 0.5f * (rParl * rParl + rPerp * rPerp);
}

// Analytic smooth RGB-to-spectral reflectance conversion across 4 wavelengths
public float4 rgbToSpectrum4(float3 rgb, float4 wavelengths)
{
    float maxVal = max(rgb.r, max(rgb.g, rgb.b));
    if (maxVal <= 1e-5f)
        return float4(0.0f);

    float3 normRgb = rgb / maxVal;
    if (abs(normRgb.r - normRgb.g) < 1e-4f && abs(normRgb.g - normRgb.b) < 1e-4f)
        return float4(normRgb.g * maxVal);

    float3 c = clamp(normRgb, 0.001f, 0.9f);
    float3 u = 2.0f * c - 1.0f;
    float3 p = u / sqrt(max(float3(1e-4f), 1.0f - u * u));

    // Exact Lagrange basis for nodes x_B = 0.20, x_G = 0.425, x_R = 0.70
    float c0 =  10.4575f * p.b - 22.2222f * p.g + 11.7647f * p.r;
    float c1 = -11.7647f * p.b + 20.0000f * p.g -  7.3529f * p.r;
    float c2 =   3.1111f * p.b -  3.1111f * p.g +  1.0000f * p.r;

    float4 x = (wavelengths - 380.0f) * (1.0f / 400.0f);
    float4 val = c0 * x * x + c1 * x + c2;
    float4 s = 0.5f + 0.5f * (val / sqrt(1.0f + val * val));
    return clamp(s, 0.0f, 1.0f) * maxVal;
}

// Wyman et al. 2013 analytical fit for CIE 1931 XYZ Color Matching Functions
public float3 cie1931XYZ(float lambda)
{
    float x1 = (lambda - 442.0f) * ((lambda < 442.0f) ? 0.0624f : 0.0374f);
    float x2 = (lambda - 599.8f) * ((lambda < 599.8f) ? 0.0264f : 0.0323f);
    float x3 = (lambda - 501.1f) * ((lambda < 501.1f) ? 0.0490f : 0.0382f);
    float x = 0.362f * exp(-0.5f * x1 * x1)
            + 1.056f * exp(-0.5f * x2 * x2)
            - 0.065f * exp(-0.5f * x3 * x3);

    float y1 = (lambda - 568.8f) * ((lambda < 568.8f) ? 0.0213f : 0.0247f);
    float y2 = (lambda - 530.9f) * ((lambda < 530.9f) ? 0.0613f : 0.0322f);
    float y = 0.821f * exp(-0.5f * y1 * y1)
            + 0.286f * exp(-0.5f * y2 * y2);

    float z1 = (lambda - 437.0f) * ((lambda < 437.0f) ? 0.0845f : 0.0278f);
    float z2 = (lambda - 459.0f) * ((lambda < 459.0f) ? 0.0385f : 0.0725f);
    float z = 1.217f * exp(-0.5f * z1 * z1)
            + 0.681f * exp(-0.5f * z2 * z2);

    return max(float3(0.0f), float3(x, y, z));
}

// Converts 4 sampled hero spectral radiance values to linear sRGB
public float3 spectral4ToLinearRgb(float4 radiance, float4 wavelengths)
{
    float3 xyz = radiance.x * cie1931XYZ(wavelengths.x)
               + radiance.y * cie1931XYZ(wavelengths.y)
               + radiance.z * cie1931XYZ(wavelengths.z)
               + radiance.w * cie1931XYZ(wavelengths.w);

    // Normalization factor (400 nm / 4 samples) / 106.86 = 0.9358f under D65 white point
    xyz *= float3(0.95047f, 1.00000f, 1.08883f) * 0.9358f;

    // Standard IEC 61966-2-1 XYZ to Linear sRGB (D65) matrix
    float3 rgb;
    rgb.r =  3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z;
    rgb.g = -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z;
    rgb.b =  0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z;

    return max(float3(0.0f), rgb);
}
