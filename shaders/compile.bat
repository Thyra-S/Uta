slangc RayGen.slang    -target spirv -profile sm_6_6 -O3 -o RayGen.spv
slangc Intersect.slang -target spirv -profile sm_6_6 -O3 -o Intersect.spv
slangc Shade.slang     -target spirv -profile sm_6_6 -O3 -o Shade.spv
slangc Tonemap.slang   -target spirv -profile sm_6_6 -O3 -o Tonemap.spv