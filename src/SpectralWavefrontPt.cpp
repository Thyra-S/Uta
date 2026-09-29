#include "SpectralWavefrontPt.h"

void SpectralWavefrontPt::run()
{
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

void SpectralWavefrontPt::initWindow()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Spectral Wavefront Path Tracer (vk_ray_query)", nullptr, nullptr);
}

void SpectralWavefrontPt::initVulkan()
{
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createCommandPool();

    // Wavefront Data Infrastructure
    createAccumulationImage();
    createWavefrontQueues();
    createScene();
    createAccelerationStructures();
    createMeshDescriptorBuffer();

    // Compute Pipeline Infrastructure
    createDescriptorSetLayout();
    createDescriptorPoolAndSets();
    createComputePipelines();

    createCommandBuffers();
    createSyncObjects();
}

/*---------- INSTANCE & DEVICE CREATION ----------*/

void SpectralWavefrontPt::createInstance()
{
    constexpr vk::ApplicationInfo appInfo{
        .pApplicationName = "Spectral Wavefront Path Tracer",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "WavefrontEngine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = vk::ApiVersion13
    };

    std::vector<const char*> requiredLayers;
    if (enableValidationLayers) {
        requiredLayers.assign(validationLayers.begin(), validationLayers.end());
    }

    auto extensions = getRequiredInstanceExtensions();

    vk::InstanceCreateInfo createInfo{
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
        .ppEnabledLayerNames = requiredLayers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data()
    };

    instance = vk::raii::Instance(context, createInfo);
}

std::vector<const char*> SpectralWavefrontPt::getRequiredInstanceExtensions()
{
    uint32_t glfwExtensionCount = 0;
    auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    if (enableValidationLayers) {
        extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }
    return extensions;
}

void SpectralWavefrontPt::pickPhysicalDevice()
{
    auto physicalDevices = instance.enumeratePhysicalDevices();
    vk::raii::PhysicalDevice chosenDevice = nullptr;
    // First pass: look for a suitable discrete GPU
    for (const auto& dev : physicalDevices) {
        if (isDeviceSuitable(dev) &&
            dev.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
            chosenDevice = dev;
            break;
        }
    }

    // Second pass fallback: take any suitable GPU (integrated) if no discrete exists
    if (!*chosenDevice) {
        auto it = std::ranges::find_if(physicalDevices, [&](const auto& dev) { return isDeviceSuitable(dev); });
        if (it != physicalDevices.end()) {
            chosenDevice = *it;
        }
    }

    if (!*chosenDevice) {
        throw std::runtime_error("No GPU found supporting Vulkan 1.3, Ray Query, and Acceleration Structures!");
    }
    physicalDevice = chosenDevice;
}

bool SpectralWavefrontPt::isDeviceSuitable(const vk::raii::PhysicalDevice& pDevice)
{
    if (pDevice.getProperties().apiVersion < vk::ApiVersion13) return false;

    // Check extensions
    auto available = pDevice.enumerateDeviceExtensionProperties();
    for (const auto& req : requiredDeviceExtensions) {
        bool found = std::ranges::any_of(available, [&](const auto& prop) {
            return strcmp(prop.extensionName, req) == 0;
            });
        if (!found) return false;
    }

    // Verify support for Ray Query, Buffer Device Address, and Acceleration Structure
    auto features2 = pDevice.template getFeatures2<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan12Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceRayQueryFeaturesKHR,
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR>();

    bool int64Supported = features2.template get < vk::PhysicalDeviceFeatures2().features.shaderInt64;
    bool bdaSupported      = features2.template get<vk::PhysicalDeviceVulkan12Features>().bufferDeviceAddress;
    bool scalarSupported   = features2.template get<vk::PhysicalDeviceVulkan12Features>().scalarBlockLayout;
    bool sync2Supported    = features2.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2;
    bool rayQuerySupported = features2.template get<vk::PhysicalDeviceRayQueryFeaturesKHR>().rayQuery;
    bool asSupported       = features2.template get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>().accelerationStructure;

    return int64Supported && bdaSupported && scalarSupported && sync2Supported && rayQuerySupported && asSupported;
}

void SpectralWavefrontPt::createLogicalDevice()
{
    auto queueProps = physicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueProps.size(); i++) 
    {
        if ((queueProps[i].queueFlags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute)) &&
            physicalDevice.getSurfaceSupportKHR(i, *surface)) 
        {
            queueIndex = i;
            break;
        }
    }

    // Modern Vulkan structure chaining to activate RT & Wavefront extensions
    vk::StructureChain<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDeviceVulkan12Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceRayQueryFeaturesKHR,
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR> featureChain{
        {.features = {.shaderInt64 = true }}, // Features2
        {.shaderDrawParameters = true},
        {
            .descriptorIndexing = true,
            .scalarBlockLayout = true,
            .bufferDeviceAddress = true // Crucial for Ray Tracing BVH and fast pointer math
        },
        {
            .synchronization2 = true,
            .dynamicRendering = true
        },
        {.rayQuery = true},             // Enables rayQueryEXT in compute shaders
        {.accelerationStructure = true} // Enables TLAS/BLAS handling
    };

    float priority = 1.0f;
    vk::DeviceQueueCreateInfo queueInfo{
        .queueFamilyIndex = queueIndex,
        .queueCount = 1,
        .pQueuePriorities = &priority
    };

    vk::DeviceCreateInfo createInfo{
        .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueInfo,
        .enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtensions.size()),
        .ppEnabledExtensionNames = requiredDeviceExtensions.data()
    };

    device = vk::raii::Device(physicalDevice, createInfo);
    computeAndGraphicsQueue = vk::raii::Queue(device, queueIndex, 0);
}

/*---------- OFFSCREEN ACCUMULATION & WAVEFRONT QUEUES ----------*/

void SpectralWavefrontPt::createAccumulationImage()
{
    // Accumulation requires 32-bit floating point precision per channel to integrate spectral radiance
    vk::ImageCreateInfo imageInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR32G32B32A32Sfloat,
        .extent = { WIDTH, HEIGHT, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | 
                vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
        .sharingMode = vk::SharingMode::eExclusive
    };

    accumulationImage = vk::raii::Image(device, imageInfo);
    auto memReq = accumulationImage.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo{
        .allocationSize = memReq.size,
        .memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
    };
    accumulationImageMemory = vk::raii::DeviceMemory(device, allocInfo);
    accumulationImage.bindMemory(*accumulationImageMemory, 0);

    vk::ImageViewCreateInfo viewInfo{
        .image = *accumulationImage,
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR32G32B32A32Sfloat,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    };
    accumulationImageView = vk::raii::ImageView(device, viewInfo);

    // One-time command buffer: transition eUndefined -> eGeneral and clear image to 0
    vk::CommandBufferAllocateInfo cmdAllocInfo{
        .commandPool = *commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };
    vk::raii::CommandBuffers cmds(device, cmdAllocInfo);
    auto& cmd = cmds[0];

    cmd.begin({ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

    vk::ImageMemoryBarrier2 initBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eNone,
        .srcAccessMask = vk::AccessFlagBits2::eNone,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .image = *accumulationImage,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &initBarrier });

    vk::ClearColorValue clearColor{ .float32 = {{ 0.0f, 0.0f, 0.0f, 0.0f }} };
    vk::ImageSubresourceRange range{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
    cmd.clearColorImage(*accumulationImage, vk::ImageLayout::eTransferDstOptimal, clearColor, range);

    vk::ImageMemoryBarrier2 toGeneralBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = *accumulationImage,
        .subresourceRange = range
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toGeneralBarrier });

    cmd.end();

    vk::SubmitInfo submitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*cmd };
    computeAndGraphicsQueue.submit(submitInfo, nullptr);
    computeAndGraphicsQueue.waitIdle();
}

void SpectralWavefrontPt::createWavefrontQueues()
{
    const uint32_t numPixels = WIDTH * HEIGHT;
    const vk::DeviceSize rayQueueSize = numPixels * sizeof(WavefrontData::RayPayload);
    const vk::DeviceSize hitQueueSize = numPixels * sizeof(WavefrontData::HitPayload);

    auto createStorageBuffer = [&](vk::DeviceSize size, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& memory) {
        vk::BufferCreateInfo bufInfo{
            .size = size,
            .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            .sharingMode = vk::SharingMode::eExclusive
        };
        buffer = vk::raii::Buffer(device, bufInfo);
        auto memReq = buffer.getMemoryRequirements();

        // Memory allocated with DEVICE_ADDRESS capability
        vk::MemoryAllocateFlagsInfo flagsInfo{ .flags = vk::MemoryAllocateFlagBits::eDeviceAddress };
        vk::MemoryAllocateInfo allocInfo{
            .pNext = &flagsInfo,
            .allocationSize = memReq.size,
            .memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
        };
        memory = vk::raii::DeviceMemory(device, allocInfo);
        buffer.bindMemory(*memory, 0);
        };

    createStorageBuffer(rayQueueSize, rayQueueBuffer, rayQueueMemory);
    createStorageBuffer(hitQueueSize, hitQueueBuffer, hitQueueMemory);
}

/*---------- SCENE, OBJECT, ACCELERATION STRUCTURE CREATION ----------*/

void SpectralWavefrontPt::createScene()
{
    meshes.clear();
    instances.clear();

    camera.position = glm::vec3(0.0f, 2.5f, 5.0f);
    camera.target = glm::vec3(0.0f, 0.5f, 0.0f);
    camera.up = glm::vec3(0.0f, 1.0f, 0.0f);
    camera.fovY = 45.0f;

    glm::mat4 view = glm::lookAt(camera.position, camera.target, camera.up);
    glm::mat4 proj = glm::perspective(
        glm::radians(camera.fovY),
        static_cast<float>(WIDTH) / static_cast<float>(HEIGHT),
        0.1f,
        1000.0f
    );

    proj[1][1] *= -1.0f;

    invView = glm::inverse(view);
    invProj = glm::inverse(proj);

    // Mesh 0: Ground Plane (Quad)
    MeshGeometry floorMesh{
         .vertices = {
             { -5.0f, 0.0f, -5.0f },
             {  5.0f, 0.0f, -5.0f },
             {  5.0f, 0.0f,  5.0f },
             { -5.0f, 0.0f,  5.0f }
         },
         .attributes = {
             { {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f} },
             { {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f} },
             { {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f} },
             { {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f} }
         },
         .indices = { 0, 1, 2, 2, 3, 0 }
    };
    meshes.push_back(std::move(floorMesh));

    // Mesh 1: Cube / Box
    MeshGeometry boxMesh{
        .vertices = {
            // Front face (Z = 0.5)
            { -0.5f, -0.5f,  0.5f }, {  0.5f, -0.5f,  0.5f }, {  0.5f,  0.5f,  0.5f }, { -0.5f,  0.5f,  0.5f },
            // Back face (Z = -0.5)
            {  0.5f, -0.5f, -0.5f }, { -0.5f, -0.5f, -0.5f }, { -0.5f,  0.5f, -0.5f }, {  0.5f,  0.5f, -0.5f },
            // Top face (Y = 0.5)
            { -0.5f,  0.5f,  0.5f }, {  0.5f,  0.5f,  0.5f }, {  0.5f,  0.5f, -0.5f }, { -0.5f,  0.5f, -0.5f },
            // Bottom face (Y = -0.5)
            { -0.5f, -0.5f, -0.5f }, {  0.5f, -0.5f, -0.5f }, {  0.5f, -0.5f,  0.5f }, { -0.5f, -0.5f,  0.5f }
        },
        .attributes = {
            // Front face
            { {0.0f, 0.0f,  1.0f}, {0.0f, 0.0f} }, { {0.0f, 0.0f,  1.0f}, {1.0f, 0.0f} },
            { {0.0f, 0.0f,  1.0f}, {1.0f, 1.0f} }, { {0.0f, 0.0f,  1.0f}, {0.0f, 1.0f} },
            // Back face
            { {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f} }, { {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f} },
            { {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f} }, { {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f} },
            // Top face
            { {0.0f, 1.0f,  0.0f}, {0.0f, 0.0f} }, { {0.0f, 1.0f,  0.0f}, {1.0f, 0.0f} },
            { {0.0f, 1.0f,  0.0f}, {1.0f, 1.0f} }, { {0.0f, 1.0f,  0.0f}, {0.0f, 1.0f} },
            // Bottom face
            { {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f} }, { {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f} },
            { {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f} }, { {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f} }
        },
        .indices = {
            0,  1,  2,  2,  3,  0, // Front
            4,  5,  6,  6,  7,  4, // Back
            8,  9, 10, 10, 11,  8, // Top
            12, 13, 14, 14, 15, 12 // Bottom
        }
    };
    meshes.push_back(std::move(boxMesh));

    // Instance 0: Floor
    instances.push_back(
        {
        .meshIndex = 0,
        .transform = glm::mat4(1.0f),
        .customInstanceId = 0,
        .materialId = 0
        }
    );

    // Instance 1: Box shifted in space
    instances.push_back(
        {
        .meshIndex = 1,
        .transform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
        .customInstanceId = 1,
        .materialId = 1
        }
    );
}

void SpectralWavefrontPt::createAccelerationStructures()
{
    for (auto& mesh : meshes)
    {
        buildBLAS(mesh);
    }

    // --- 3. Build TLAS ---
    buildTLAS();
}

void SpectralWavefrontPt::buildBLAS(MeshGeometry& mesh)
{
    auto createBufferWithAddress = [&](vk::DeviceSize size, vk::BufferUsageFlags usage,
        vk::raii::Buffer& buffer, vk::raii::DeviceMemory& memory)
        {
            vk::BufferCreateInfo info{
                .size = size,
                .usage = usage | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                .sharingMode = vk::SharingMode::eExclusive
            };
            buffer = vk::raii::Buffer(device, info);
            auto memReq = buffer.getMemoryRequirements();

            vk::MemoryAllocateFlagsInfo flagsInfo{ .flags = vk::MemoryAllocateFlagBits::eDeviceAddress };
            vk::MemoryAllocateInfo allocInfo{
                .pNext = &flagsInfo,
                .allocationSize = memReq.size,
                .memoryTypeIndex = findMemoryType(memReq.memoryTypeBits,
                                                  vk::MemoryPropertyFlagBits::eHostVisible |
                                                  vk::MemoryPropertyFlagBits::eHostCoherent)
            };
            memory = vk::raii::DeviceMemory(device, allocInfo);
            buffer.bindMemory(*memory, 0);
        };

    const vk::BufferUsageFlags geomUsage = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
        vk::BufferUsageFlagBits::eStorageBuffer;

    const vk::DeviceSize vSize = mesh.vertices.size() * sizeof(glm::vec3);
    const vk::DeviceSize aSize = mesh.attributes.size() * sizeof(VertexAttributes);
    const vk::DeviceSize iSize = mesh.indices.size() * sizeof(uint32_t);

    createBufferWithAddress(vSize, geomUsage, mesh.vertexBuffer, mesh.vertexBufferMemory);
    createBufferWithAddress(aSize, vk::BufferUsageFlagBits::eStorageBuffer, mesh.attributeBuffer, mesh.attributeBufferMemory);
    createBufferWithAddress(iSize, geomUsage, mesh.indexBuffer, mesh.indexBufferMemory);

    void* vData = mesh.vertexBufferMemory.mapMemory(0, vSize);
    memcpy(vData, mesh.vertices.data(), vSize);
    mesh.vertexBufferMemory.unmapMemory();

    void* aData = mesh.attributeBufferMemory.mapMemory(0, aSize);
    memcpy(aData, mesh.attributes.data(), aSize);
    mesh.attributeBufferMemory.unmapMemory();

    void* iData = mesh.indexBufferMemory.mapMemory(0, iSize);
    memcpy(iData, mesh.indices.data(), iSize);
    mesh.indexBufferMemory.unmapMemory();

    vk::DeviceAddress vertexAddress = device.getBufferAddress({ .buffer = *mesh.vertexBuffer });
    vk::DeviceAddress indexAddress = device.getBufferAddress({ .buffer = *mesh.indexBuffer });

    vk::AccelerationStructureGeometryTrianglesDataKHR triangles{
        .vertexFormat = vk::Format::eR32G32B32Sfloat,
        .vertexData = {.deviceAddress = vertexAddress },
        .vertexStride = sizeof(glm::vec3),
        .maxVertex = static_cast<uint32_t>(mesh.vertices.size()),
        .indexType = vk::IndexType::eUint32,
        .indexData = {.deviceAddress = indexAddress }
    };

    vk::AccelerationStructureGeometryKHR asGeom{
        .geometryType = vk::GeometryTypeKHR::eTriangles,
        .geometry = {.triangles = triangles },
        .flags = vk::GeometryFlagBitsKHR::eOpaque
    };

    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{
        .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = 1,
        .pGeometries = &asGeom
    };

    uint32_t primitiveCount = static_cast<uint32_t>(mesh.indices.size() / 3);
    auto buildSizes = device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo, { primitiveCount });

    createBufferWithAddress(buildSizes.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR,
        mesh.blasBuffer, mesh.blasMemory);

    vk::AccelerationStructureCreateInfoKHR createInfo{
        .buffer = *mesh.blasBuffer,
        .size = buildSizes.accelerationStructureSize,
        .type = vk::AccelerationStructureTypeKHR::eBottomLevel
    };
    mesh.blas = vk::raii::AccelerationStructureKHR(device, createInfo);

    // Build the BLAS on GPU
    buildAccelerationStructureOnGPU(buildInfo, *mesh.blas, buildSizes.buildScratchSize, primitiveCount);

    // Save BLAS 64-bit device address for TLAS instances
    vk::AccelerationStructureDeviceAddressInfoKHR addressInfo{ .accelerationStructure = *mesh.blas };
    mesh.blasAddress = device.getAccelerationStructureAddressKHR(addressInfo);
}

void SpectralWavefrontPt::buildTLAS()
{
    std::vector<vk::AccelerationStructureInstanceKHR> geometryInstances;
    geometryInstances.reserve(instances.size());

    for (const auto& inst : instances) {
        // Convert glm::mat4 (column-major) to VkTransformMatrixKHR (3x4 row-major)
        vk::TransformMatrixKHR transformMatrix{};
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 4; ++col) {
                transformMatrix.matrix[row][col] = inst.transform[col][row];
            }
        }

        vk::AccelerationStructureInstanceKHR asInstance{
            .transform = transformMatrix,
            .instanceCustomIndex = inst.customInstanceId,
            .mask = 0xFF,
            .instanceShaderBindingTableRecordOffset = 0,
            .flags = static_cast<VkGeometryInstanceFlagsKHR>(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable),
            .accelerationStructureReference = meshes[inst.meshIndex].blasAddress
        };
        geometryInstances.push_back(asInstance);
    }

    const vk::DeviceSize instancesBufferSize = geometryInstances.size() * sizeof(vk::AccelerationStructureInstanceKHR);

    // Staging / Device buffer for instance descriptions
    vk::BufferCreateInfo instanceBufInfo{
        .size = instancesBufferSize,
        .usage = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        .sharingMode = vk::SharingMode::eExclusive
    };
    vk::raii::Buffer instanceBuffer(device, instanceBufInfo);
    auto instMemReq = instanceBuffer.getMemoryRequirements();

    vk::MemoryAllocateFlagsInfo flagsInfo{ .flags = vk::MemoryAllocateFlagBits::eDeviceAddress };
    vk::MemoryAllocateInfo allocInfo{
        .pNext = &flagsInfo,
        .allocationSize = instMemReq.size,
        .memoryTypeIndex = findMemoryType(instMemReq.memoryTypeBits,
                                          vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
    };
    vk::raii::DeviceMemory instanceMemory(device, allocInfo);
    instanceBuffer.bindMemory(*instanceMemory, 0);

    void* data = instanceMemory.mapMemory(0, instancesBufferSize);
    memcpy(data, geometryInstances.data(), instancesBufferSize);
    instanceMemory.unmapMemory();

    vk::DeviceAddress instanceBufferAddress = device.getBufferAddress({ .buffer = *instanceBuffer });

    vk::AccelerationStructureGeometryInstancesDataKHR instancesData{
        .arrayOfPointers = false,
        .data = {.deviceAddress = instanceBufferAddress }
    };

    vk::AccelerationStructureGeometryKHR topAsGeometry{
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry = {.instances = instancesData }
    };

    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = 1,
        .pGeometries = &topAsGeometry
    };

    uint32_t primitiveCount = static_cast<uint32_t>(geometryInstances.size());
    auto buildSizes = device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo, { primitiveCount });

    // Allocate TLAS destination buffer
    vk::BufferCreateInfo tlasBufInfo{
        .size = buildSizes.accelerationStructureSize,
        .usage = vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        .sharingMode = vk::SharingMode::eExclusive
    };
    tlasBuffer = vk::raii::Buffer(device, tlasBufInfo);
    auto tlasMemReq = tlasBuffer.getMemoryRequirements();

    vk::MemoryAllocateInfo tlasAllocInfo{
        .pNext = &flagsInfo,
        .allocationSize = tlasMemReq.size,
        .memoryTypeIndex = findMemoryType(tlasMemReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
    };
    tlasMemory = vk::raii::DeviceMemory(device, tlasAllocInfo);
    tlasBuffer.bindMemory(*tlasMemory, 0);

    vk::AccelerationStructureCreateInfoKHR createInfo{
        .buffer = *tlasBuffer,
        .size = buildSizes.accelerationStructureSize,
        .type = vk::AccelerationStructureTypeKHR::eTopLevel
    };
    tlas = vk::raii::AccelerationStructureKHR(device, createInfo);

    buildAccelerationStructureOnGPU(buildInfo, *tlas, buildSizes.buildScratchSize, primitiveCount);

    vk::AccelerationStructureDeviceAddressInfoKHR addressInfo{ .accelerationStructure = *tlas };
    tlasDeviceAddress = device.getAccelerationStructureAddressKHR(addressInfo);
}

void SpectralWavefrontPt::buildAccelerationStructureOnGPU(
    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo,
    vk::AccelerationStructureKHR as,
    vk::DeviceSize scratchSize,
    uint32_t primitiveCount)
{
    // 1. Allocate scratch buffer with Device Address support
    vk::BufferCreateInfo scratchInfo{
        .size = scratchSize,
        .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        .sharingMode = vk::SharingMode::eExclusive
    };
    vk::raii::Buffer scratchBuffer(device, scratchInfo);
    auto memReq = scratchBuffer.getMemoryRequirements();

    vk::MemoryAllocateFlagsInfo flagsInfo{ .flags = vk::MemoryAllocateFlagBits::eDeviceAddress };
    vk::MemoryAllocateInfo allocInfo{
        .pNext = &flagsInfo,
        .allocationSize = memReq.size,
        .memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
    };
    vk::raii::DeviceMemory scratchMemory(device, allocInfo);
    scratchBuffer.bindMemory(*scratchMemory, 0);

    vk::DeviceAddress scratchAddress = device.getBufferAddress({ .buffer = *scratchBuffer });

    buildInfo.dstAccelerationStructure = as;
    buildInfo.scratchData.deviceAddress = scratchAddress;

    // 2. Build via a one-time command buffer
    vk::CommandBufferAllocateInfo cmdAllocInfo{
        .commandPool = *commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };
    vk::raii::CommandBuffers cmds(device, cmdAllocInfo);
    auto& cmd = cmds[0];

    cmd.begin({ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

    vk::AccelerationStructureBuildRangeInfoKHR rangeInfo{
        .primitiveCount = primitiveCount,
        .primitiveOffset = 0,
        .firstVertex = 0,
        .transformOffset = 0
    };
    const vk::AccelerationStructureBuildRangeInfoKHR* pRangeInfos = &rangeInfo;

    cmd.buildAccelerationStructuresKHR(buildInfo, pRangeInfos);

    // Memory barrier to guarantee the build finishes before subsequent access
    vk::MemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
        .srcAccessMask = vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
        .dstStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR | vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eShaderRead
    };
    vk::DependencyInfo depInfo{ .memoryBarrierCount = 1, .pMemoryBarriers = &barrier };
    cmd.pipelineBarrier2(depInfo);

    cmd.end();

    vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &*cmd
    };
    computeAndGraphicsQueue.submit(submitInfo, nullptr);
    computeAndGraphicsQueue.waitIdle();
}

void SpectralWavefrontPt::createMeshDescriptorBuffer()
{
    // Build host array of descriptors referencing each instance's mesh buffers
    std::vector<WavefrontData::GpuMeshDesc> descriptors;
    descriptors.reserve(instances.size());

    for (const auto& inst : instances) {
        const auto& mesh = meshes[inst.meshIndex];
        descriptors.push_back(WavefrontData::GpuMeshDesc
            {
            .positionBufferAddress = device.getBufferAddress({.buffer = *mesh.vertexBuffer }),
            .attributtesBufferAddress = device.getBufferAddress({.buffer = *mesh.attributeBuffer}),
            .indexBufferAddress = device.getBufferAddress({.buffer = *mesh.indexBuffer }),
            .materialId = inst.materialId,
            .padding = 0
            }
        );
    }

    vk::DeviceSize bufferSize = sizeof(WavefrontData::GpuMeshDesc) * descriptors.size();

    // Allocate host-visible GPU buffer with device address support
    vk::BufferCreateInfo bufInfo
    {
        .size = bufferSize,
        .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        .sharingMode = vk::SharingMode::eExclusive
    };
    meshDescBuffer = vk::raii::Buffer(device, bufInfo);
    auto memReq = meshDescBuffer.getMemoryRequirements();

    vk::MemoryAllocateFlagsInfo flagsInfo{ .flags = vk::MemoryAllocateFlagBits::eDeviceAddress };
    vk::MemoryAllocateInfo allocInfo
    {
        .pNext = &flagsInfo,
        .allocationSize = memReq.size,
        .memoryTypeIndex = findMemoryType(memReq.memoryTypeBits,
                                          vk::MemoryPropertyFlagBits::eHostVisible |
                                          vk::MemoryPropertyFlagBits::eHostCoherent)
    };
    meshDescMemory = vk::raii::DeviceMemory(device, allocInfo);
    meshDescBuffer.bindMemory(*meshDescMemory, 0);

    // Copy descriptor table to GPU
    void* data = meshDescMemory.mapMemory(0, bufferSize);
    memcpy(data, descriptors.data(), bufferSize);
    meshDescMemory.unmapMemory();

    // Query 64-bit device address to pass in Push Constants
    meshDescAddress = device.getBufferAddress({ .buffer = *meshDescBuffer });
}

/*---------- DESCRIPTORS & COMPUTE PIPELINES ----------*/

void SpectralWavefrontPt::createDescriptorSetLayout()
{
    // Binding 0: Top-Level Acceleration Structure (TLAS) accessed via Ray Query
    // Binding 1: Offscreen HDR Accumulation Image (RW Storage Image)
    // Binding 2: Ray Queue (SSBO)
    // Binding 3: Hit Queue (SSBO)
    std::array<vk::DescriptorSetLayoutBinding, 4> bindings{ {
        {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute
        },
        {
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute
        },
        {
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute
        },
        {
            .binding = 3,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute
        }
    } };

    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()
    };
    descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

    // Global Push Constants
    vk::PushConstantRange pushConstantRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(WavefrontData::SceneParams)
    };

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &*descriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange
    };
    computePipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);
}

void SpectralWavefrontPt::createDescriptorPoolAndSets()
{
    std::array<vk::DescriptorPoolSize, 3> poolSizes{ {
        { vk::DescriptorType::eAccelerationStructureKHR, 1 },
        { vk::DescriptorType::eStorageImage, 1 },
        { vk::DescriptorType::eStorageBuffer, 2 }
    } };

    vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 1,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };
    descriptorPool = vk::raii::DescriptorPool(device, poolInfo);

    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &*descriptorSetLayout
    };
    descriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    // Acceleration structure write descriptor
    vk::WriteDescriptorSetAccelerationStructureKHR asInfo{
        .accelerationStructureCount = 1,
        .pAccelerationStructures = &*tlas
    };

    // Bind Buffers and Images to Descriptor Sets
    vk::DescriptorImageInfo accumImageInfo{
        .imageView = *accumulationImageView,
        .imageLayout = vk::ImageLayout::eGeneral
    };
    vk::DescriptorBufferInfo rayBufferInfo{
        .buffer = *rayQueueBuffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE
    };
    vk::DescriptorBufferInfo hitBufferInfo{
        .buffer = *hitQueueBuffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE
    };

    std::array<vk::WriteDescriptorSet, 4> descriptorWrites{ {
        {
            .pNext = &asInfo,
            .dstSet = *descriptorSets[0],
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eAccelerationStructureKHR
        },
        {
            .dstSet = *descriptorSets[0],
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &accumImageInfo
        },
        {
            .dstSet = *descriptorSets[0],
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &rayBufferInfo
        },
        {
            .dstSet = *descriptorSets[0],
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &hitBufferInfo
        }
    } };

    device.updateDescriptorSets(descriptorWrites, nullptr);
}

void SpectralWavefrontPt::createComputePipelines()
{
    // Local lambda function to create the different compute pipeline stages.
    auto buildPipeline = [&](vk::raii::Pipeline& outPipeline, const std::string& path) {
        auto code = readFile(path);
        vk::raii::ShaderModule module = createShaderModule(code);

        vk::ComputePipelineCreateInfo info{
            .stage = {
                .stage = vk::ShaderStageFlagBits::eCompute,
                .module = *module,
                .pName = "main"
            },
            .layout = *computePipelineLayout
        };

        outPipeline = vk::raii::Pipeline(device, nullptr, info);
        };
    
    buildPipeline(pipelineRayGen, "shaders/RayGen.spv");
    buildPipeline(pipelineIntersect, "shaders/Intersect.spv");
    buildPipeline(pipelineShade, "shaders/Shade.spv");
    buildPipeline(pipelineTonemap, "shaders/Tonemap.spv");
}

/*---------- COMMAND BUFFER: WAVEFRONT DISPATCH LOOP ----------*/

void SpectralWavefrontPt::recordWavefrontCommands(uint32_t imageIndex)
{
    auto& cmd = commandBuffers[frameIndex];
    cmd.begin({});

    WavefrontData::SceneParams params{
        .invView = invView,
        .invProj = invProj,
        .vertexBufferAddress = meshDescAddress,
        .indexBufferAddress = 0,
        .shiftOffset = glm::vec2(0.0f),
        .tiltAxis = glm::vec2(0.0f, 1.0f),
        .tiltAngle = 0.0f,
        .focalDistance = 5.0f,
        .apertureRadius = 0.0f,
        .width = WIDTH,
        .height = HEIGHT,
        .frameCounter = frameIndex,
        .currentBounce = 0,
        .maxBounces = 8,
        .totalPixels = WIDTH * HEIGHT
    };

    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 0, *descriptorSets[0], nullptr);

    const uint32_t workgroupsX = (WIDTH + 15) / 16;
    const uint32_t workgroupsY = (HEIGHT + 15) / 16;
    const uint32_t totalPixels = WIDTH * HEIGHT;
    const uint32_t wavefrontWorkgroups = (totalPixels + 63) / 64;

    // --- PHASE 1: RAY GENERATION ---
    // Generates camera rays, samples 4 spectral wavelengths per sample per pixel, populates RayQueue
    if (*pipelineRayGen) 
    {
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelineRayGen);
        cmd.pushConstants<WavefrontData::SceneParams>(*computePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, params);
        cmd.dispatch(wavefrontWorkgroups, 1, 1);
    }

    // Barrier: Ensure all rays are written to RayQueue before traversal
    vk::MemoryBarrier2 rayGenBarrier
    {
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead
    };
    vk::DependencyInfo depInfo{ .memoryBarrierCount = 1, .pMemoryBarriers = &rayGenBarrier };
    cmd.pipelineBarrier2(depInfo);

    // --- PHASE 2 & 3: WAVEFRONT BOUNCE LOOP ---
    for (uint32_t bounce = 0; bounce < params.maxBounces; bounce++)
    {
        params.currentBounce = bounce;
        cmd.pushConstants<WavefrontData::SceneParams>(*computePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, params);

        // 2a. Intersect: Traces ray queries against TLAS, fills HitQueue
        if (*pipelineIntersect) 
        {
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelineIntersect);
            cmd.dispatch(wavefrontWorkgroups, 1, 1);
        }

        // Barrier: Wait for Intersect hits before shading
        vk::MemoryBarrier2 intersectBarrier
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead
        };
        depInfo.pMemoryBarriers = &intersectBarrier;
        cmd.pipelineBarrier2(depInfo);

        // 2b. Spectral Shade: Evaluates BSDF, Hero wavelength sampling, writes radiance to AccumulationImage
        if (*pipelineShade) 
        {
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelineShade);
            cmd.dispatch(wavefrontWorkgroups, 1, 1);
        }

        // Barrier: Wait for Shade writes to RayQueue before next bounce traversal
        vk::MemoryBarrier2 shadeBarrier
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead
        };
        depInfo.pMemoryBarriers = &shadeBarrier;
        cmd.pipelineBarrier2(depInfo);
    }

    // --- PHASE 4: COMPOSITE / TONEMAP & PRESENTATION ---
    // In a production engine, blit or run a final tone-mapping compute pass converting XYZ to sRGB onto swapChainImages[imageIndex]

    // 1. Run Tone Mapping & XYZ->RGB Conversion Pass
    if (*pipelineTonemap) 
    {
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelineTonemap);
        cmd.pushConstants<WavefrontData::SceneParams>(*computePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, params);
        cmd.dispatch(workgroupsX, workgroupsY, 1);
    }

    vk::ImageMemoryBarrier2 accumToSrcBarrier
    {
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eTransferSrcOptimal,
        .image = *accumulationImage,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    };

    vk::ImageMemoryBarrier2 swapchainToDstBarrier
    {
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eNone,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .image = swapChainImages[imageIndex],
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    };

    std::array<vk::ImageMemoryBarrier2, 2> preBlitBarriers = { accumToSrcBarrier, swapchainToDstBarrier };
    cmd.pipelineBarrier2(vk::DependencyInfo
        {
        .imageMemoryBarrierCount = static_cast<uint32_t>(preBlitBarriers.size()),
        .pImageMemoryBarriers = preBlitBarriers.data()
        });

    // 3. Blit from HDR Float Accumulation Image to SDR sRGB Swapchain Image
    vk::ImageBlit blitRegion
    {
        .srcSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 },
        .srcOffsets = {{
            vk::Offset3D{ 0, 0, 0 },
            vk::Offset3D{ static_cast<int32_t>(WIDTH), static_cast<int32_t>(HEIGHT), 1 }
        }},
        .dstSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 },
        .dstOffsets = {{
            vk::Offset3D{ 0, 0, 0 },
            vk::Offset3D{ static_cast<int32_t>(WIDTH), static_cast<int32_t>(HEIGHT), 1 }
        }}
    };

    cmd.blitImage(
        *accumulationImage, vk::ImageLayout::eTransferSrcOptimal,
        swapChainImages[imageIndex], vk::ImageLayout::eTransferDstOptimal,
        { blitRegion }, vk::Filter::eNearest
    );

    // 4. Transition Swapchain: Transfer Dst -> Present Src
    vk::ImageMemoryBarrier2 swapchainToPresentBarrier
    {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eNone,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::ePresentSrcKHR,
        .image = swapChainImages[imageIndex],
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    };

    // 5. Restore Accumulation Image: Transfer Src -> General (for the next frame)
    vk::ImageMemoryBarrier2 restoreAccumBarrier
    {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = *accumulationImage,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    };

    std::array<vk::ImageMemoryBarrier2, 2> postBlitBarriers = { restoreAccumBarrier, swapchainToPresentBarrier };
    cmd.pipelineBarrier2(vk::DependencyInfo
        {
        .imageMemoryBarrierCount = static_cast<uint32_t>(postBlitBarriers.size()),
        .pImageMemoryBarriers = postBlitBarriers.data()
        });

    cmd.end();
}

/*---------- SWAPCHAIN, COMMANDS, AND RUNTIME BOILERPLATE ----------*/

void SpectralWavefrontPt::createSwapChain()
{
    auto caps = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    swapChainExtent = { WIDTH, HEIGHT };
    swapChainSurfaceFormat = { vk::Format::eB8G8R8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear };

    vk::SwapchainCreateInfoKHR createInfo
    {
        .surface = *surface,
        .minImageCount = std::max(3u, caps.minImageCount),
        .imageFormat = swapChainSurfaceFormat.format,
        .imageColorSpace = swapChainSurfaceFormat.colorSpace,
        .imageExtent = swapChainExtent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst,
        .imageSharingMode = vk::SharingMode::eExclusive,
        .preTransform = caps.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = vk::PresentModeKHR::eMailbox,
        .clipped = true
    };

    swapChain = vk::raii::SwapchainKHR(device, createInfo);
    swapChainImages = swapChain.getImages();
}

void SpectralWavefrontPt::createImageViews()
{
    swapChainImageViews.clear();
    vk::ImageViewCreateInfo viewInfo
    {
        .viewType = vk::ImageViewType::e2D,
        .format = swapChainSurfaceFormat.format,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    };
    for (auto img : swapChainImages) 
    {
        viewInfo.image = img;
        swapChainImageViews.emplace_back(device, viewInfo);
    }
}

void SpectralWavefrontPt::createCommandPool()
{
    vk::CommandPoolCreateInfo poolInfo
    {
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueIndex
    };
    commandPool = vk::raii::CommandPool(device, poolInfo);
}

void SpectralWavefrontPt::createCommandBuffers()
{
    vk::CommandBufferAllocateInfo allocInfo
    {
        .commandPool = *commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = MAX_FRAMES_IN_FLIGHT
    };
    commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
}

void SpectralWavefrontPt::createSyncObjects()
{
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
    {
        presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});
        renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});
        inFlightFences.emplace_back(device, vk::FenceCreateInfo{ .flags = vk::FenceCreateFlagBits::eSignaled });
    }
}

void SpectralWavefrontPt::drawFrame()
{
    auto res = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
    device.resetFences(*inFlightFences[frameIndex]);

    auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);

    commandBuffers[frameIndex].reset({});
    recordWavefrontCommands(imageIndex);

    vk::PipelineStageFlags waitMask(vk::PipelineStageFlagBits::eComputeShader);
    vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*presentCompleteSemaphores[frameIndex],
        .pWaitDstStageMask = &waitMask,
        .commandBufferCount = 1,
        .pCommandBuffers = &*commandBuffers[frameIndex],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*renderFinishedSemaphores[frameIndex]
    };

    computeAndGraphicsQueue.submit(submitInfo, *inFlightFences[frameIndex]);

    vk::PresentInfoKHR presentInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*renderFinishedSemaphores[frameIndex],
        .swapchainCount = 1,
        .pSwapchains = &*swapChain,
        .pImageIndices = &imageIndex
    };

    result = computeAndGraphicsQueue.presentKHR(presentInfo);
    frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
}

void SpectralWavefrontPt::mainLoop()
{
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        drawFrame();
    }
    device.waitIdle();
}

uint32_t SpectralWavefrontPt::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
{
    auto memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

std::vector<char> SpectralWavefrontPt::readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Failed to open file: " + filename);
    std::vector<char> buffer(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    return buffer;
}

vk::raii::ShaderModule SpectralWavefrontPt::createShaderModule(const std::vector<char>& code) const
{
    vk::ShaderModuleCreateInfo createInfo{
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t*>(code.data())
    };
    return vk::raii::ShaderModule(device, createInfo);
}

void SpectralWavefrontPt::createSurface()
{
    VkSurfaceKHR s;
    if (glfwCreateWindowSurface(*instance, window, nullptr, &s) != 0) {
        throw std::runtime_error("Failed to create surface");
    }
    surface = vk::raii::SurfaceKHR(instance, s);
}

void SpectralWavefrontPt::cleanupSwapChain()
{
    swapChainImageViews.clear();
    swapChain = nullptr;
}

void SpectralWavefrontPt::recreateSwapChain()
{
    device.waitIdle();
    cleanupSwapChain();
    createSwapChain();
    createImageViews();
}

void SpectralWavefrontPt::cleanup()
{
    device.waitIdle();
    cleanupSwapChain();
    glfwDestroyWindow(window);
    glfwTerminate();
}

void SpectralWavefrontPt::setupDebugMessenger()
{
    if (!enableValidationLayers) return;
    vk::DebugUtilsMessengerCreateInfoEXT createInfo{
        .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        .pfnUserCallback = &debugCallback
    };
    debugMessenger = instance.createDebugUtilsMessengerEXT(createInfo);
}

VKAPI_ATTR vk::Bool32 VKAPI_CALL SpectralWavefrontPt::debugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT,
    vk::DebugUtilsMessageTypeFlagsEXT,
    const vk::DebugUtilsMessengerCallbackDataEXT* pData,
    void*)
{
    std::cerr << "Validation: " << pData->pMessage << std::endl;
    return vk::False;
}

int main()
{
    try {
        SpectralWavefrontPt app;
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}