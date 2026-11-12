// OpenPBRMat.slang
module OpenPbrMaterial;

public struct OpenPbrMaterial
{
    // Base Lobe
    // Spectral polynomial coefficients: xyz = (c0, c1, c2), w = max scale
    public float4 baseColorCoeffs;
    public float  baseWeight;
    public float  baseRoughness;
    public float  baseMetalness;
    public float  baseDiffuseRoughness;

    // Specular Lobe
    // Specular tint / conductor edge tint: xyz = (c0, c1, c2), w = max scale
    public float4 specularColorCoeffs;
    public float  specularWeight;
    public float  specularRoughness;
    public float  specularIor;
    public float  specularRoughnessAnisotropy;

    // Transmission Lobe
    public float4 transmissionColorCoeffs;
    // xyz = volume scattering coefficients, w = Abbe number / dispersion coefficient
    public float4 transmissionScatterAndDispersion;
    public float  transmissionWeight;
    public float  transmissionDepth;
    public float  padTrans0;
    public float  padTrans1;

    // Subsurface Lobe
    // xyz = subsurface albedo / tint coeffs, w = weight
    public float4 subsurfaceColorAndWeight;
    // xyz = mean free path (radius), w = scale
    public float4 subsurfaceRadiusAndScale;

    // Coat Lobe
    // xyz = coat color / absorption tint coeffs, w = darkening factor
    public float4 coatColorAndDarkening;
    public float  coatWeight;
    public float  coatRoughness;
    public float  coatIor;
    public float  padCoat;

    // Emission & Geometry
    // xyz = emission color spectrum, w = luminance (nits / cd/m^2)
    public float4 emissionAndLuminance;

    // Texture indices into bindless descriptor arrays (~0u if unbound)
    public uint   baseColorTexId;
    public uint   specularRoughnessTexId;
    public uint   coatRoughnessTexId;
    public uint   normalTexId;

    public uint   thinWalled;
    public uint   pad0;
    public uint   pad1;
    public uint   pad2;
};
