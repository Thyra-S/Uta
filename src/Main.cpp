#include "SpectralWavefrontPt.h"

int main(int argc, char* argv[])
{
    std::string modelPath = (argc > 1) ? argv[1] : "models/utah_teapot.usda";
    std::string materialPath = (argc > 2) ? argv[2] : "";
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