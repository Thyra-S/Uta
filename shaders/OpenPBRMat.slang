module OpenPbrMaterial;

public struct OpenPbrMaterial
{
    /* ------------------BASE LOBE------------------ */
    // Spectral polynomial coefficients: xyz = (c0, c1, c2), w = max scale
    public float4 baseColorCoeffs;
    public float baseWeight;
    public float baseRoughness;
    public float baseMetalness;
    public float baseDiffuseRoughness;

    /* ------------------ SPECULAR LOBE ------------------ */
    // Specular tint / conductor edge tint: xyz = (c0, c1, c2), w = max scale
    public float4 specularColorCoeffs;
    public float specularWeight;
    public float specularRoughness;
    public float specularIor;
    public float specularRoughnessAnisotropy;

    /* ------------------TRANSMISSION LOBE------------------ */
    public float4 transmissionColorCoeffs;
    // xyz = volume scattering coefficients, w = Abbe number / dispersion coefficient
    public float4 transmissionScatterAndDispersion;
    public float transmissionWeight;
    public float transmissionDepth;
    public float padTrans0;
    public float padTrans1;

    /* ------------------SUBSURFACE LOBE------------------ */
    // xyz = subsurface albedo / tint coeffs, w = weight
    public float4 subsurfaceColorAndWeight;
    // xyz = mean free path (radius), w = scale
    public float4 subsurfaceRadiusAndScale;

    /* ------------------COAT LOBE------------------ */
    // xyz = coat color / absorption tint coeffs, w = darkening factor
    public float4 coatColorAndDarkening;
    public float coatWeight;
    public float coatRoughness;
    public float coatIor;
    public float padCoat;

    /* ------------------ EMISSION & GEOMETRY ------------------ */
    // xyz = emission color spectrum, w = luminance (nits / cd/m^2)
    public float4 emissionAndLuminance;

    // Texture indices into bindless descriptor arrays (~0u if unbound)
    public uint baseColorTexId;
    public uint specularRoughnessTexId;
    public uint coatRoughnessTexId;
    public uint normalTexId;

    public uint thinWalled;
    public uint pad0;
    public uint pad1;
    public uint pad2;

    /* ------------------ HELPER METHODS ------------------ */

    // Evaluates a bounded sigmoid polynomial: f(lambda) in [0, coeffs.w]
    // Normalized lambda in range [-1.0, 1.0] across 380nm to 780nm
    public static float4 evalSpectralPolynomial(float4 coeffs, float4 lambda)
    {
        float4 normLambda = (lambda - 380.0) / 400.0 * 2.0 - 1.0;
        float4 x = coeffs.x * normLambda * normLambda + coeffs.y * normLambda + coeffs.z;
        return coeffs.w * (0.5 + 0.5 * (x / sqrt(1.0 + x * x)));
    }

    // Evaluates 4-bin spectral reflectance for the base diffuse/metal lobe
    public float4 getBaseReflectance(float4 lambda)
    {
        return evalSpectralPolynomial(baseColorCoeffs, lambda) * baseWeight;
    }

    // Evaluates 4-bin spectral tint for dielectric specular reflection or conductor edge tint
    public float4 getSpecularColor(float4 lambda)
    {
        return evalSpectralPolynomial(specularColorCoeffs, lambda) * specularWeight;
    }

    // Evaluates 4-bin spectral transmittance
    public float4 getTransmissionColor(float4 lambda)
    {
        return evalSpectralPolynomial(transmissionColorCoeffs, lambda) * transmissionWeight;
    }

    // Evaluates emission radiance in W/(m^2 * sr * nm) across hero wavelength bins
    public float4 getEmission(float4 lambda)
    {
        return emissionAndLuminance.xyz * emissionAndLuminance.w;
    }

    // Evaluates chromatic dispersion (Sellmeier/Cauchy) for dielectric IOR
    public float4 getDispersedIor(float4 lambda)
    {
        float dispersion = transmissionScatterAndDispersion.w;
        if (dispersion <= 0.0)
        {
            return float4(specularIor);
        }
        // Simplified Cauchy approximation: n(lambda) = n0 + B / lambda^2 (lambda in micrometers)
        float4 lambdaMicrons = lambda * 1e-3;
        return float4(specularIor) + dispersion / (lambdaMicrons * lambdaMicrons);
    }
};

struct SpectralWeights<let N : int> 
{
    float values[N];

    [mutating]
    void operator*=(SpectralWeights<N> rhs) 
    {
        for (int i = 0; i < N; ++i) values[i] *= rhs.values[i];
    }
};

interface IBSDF 
{
    SpectralWeights<4> eval(float3 wo, float3 wi, float4 lambda);
    bool sample(float3 wo, out float3 wi, out float pdf, float4 lambda, inout uint seed);
};

struct OpenPbrSurface : IBSDF 
{
    OpenPbrMaterial mat;

    SpectralWeights<4> eval(float3 wo, float3 wi, float4 lambda) 
    {
        return {};
    }

    bool sample(float3 wo, out float3 wi, out float pdf, float4 lambda, inout uint seed) 
    {
        return true;
    }
};