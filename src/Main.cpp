#include "SpectralWavefrontPt.h"

int main(int argc, char* argv[])
{
    std::string modelPath = (argc > 1) ? argv[1] : "models/scene.usd";
    std::string materialPath = (argc > 2) ? argv[2] : "materials/shader.mtlx";
    try {
        SpectralWavefrontPt app;
        app.run(modelPath, materialPath);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}