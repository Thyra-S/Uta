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
    createAccelerationStructuresPlaceholder();

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
    auto it = std::ranges::find_if(physicalDevices, [&](const auto& dev) { return isDeviceSuitable(dev); });
    if (it == physicalDevices.end()) {
        throw std::runtime_error("No GPU found supporting Vulkan 1.3, Ray Query, and Acceleration Structures!");
    }
    physicalDevice = *it;
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

    bool bdaSupported = features2.template get<vk::PhysicalDeviceVulkan12Features>().bufferDeviceAddress;
    bool sync2Supported = features2.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2;
    bool rayQuerySupported = features2.template get<vk::PhysicalDeviceRayQueryFeaturesKHR>().rayQuery;
    bool asSupported = features2.template get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>().accelerationStructure;

    return bdaSupported && sync2Supported && rayQuerySupported && asSupported;
}

void SpectralWavefrontPt::createLogicalDevice()
{
    auto queueProps = physicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueProps.size(); i++) {
        if ((queueProps[i].queueFlags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute)) &&
            physicalDevice.getSurfaceSupportKHR(i, *surface)) {
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
        {}, // Features2
        {.shaderDrawParameters = true},
        {
            .descriptorIndexing = true,
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
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc,
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

void SpectralWavefrontPt::createAccelerationStructuresPlaceholder()
{
    // Scaffolding handles for TLAS / BLAS. 
    // In actual usage, create BLAS using triangle geometric data with vk::AccelerationStructureGeometryKHR,
    // then query scratch size requirements and build via cmdBuildAccelerationStructuresKHR.
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

    std::array<vk::WriteDescriptorSet, 3> descriptorWrites{ {
        {
            .dstSet = *descriptorSets[0],
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &accumImageInfo
        },
        {
            .dstSet = *descriptorSets[0],
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &rayBufferInfo
        },
        {
            .dstSet = *descriptorSets[0],
            .dstBinding = 3,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &hitBufferInfo
        }
    } };

    device.updateDescriptorSets(descriptorWrites, nullptr);
}

void SpectralWavefrontPt::createComputePipelines()
{
    auto buildStage = [&](const std::string& path) {
        auto code = readFile(path);
        vk::raii::ShaderModule module = createShaderModule(code);
        return vk::ComputePipelineCreateInfo{
            .stage = {
                .stage = vk::ShaderStageFlagBits::eCompute,
                .module = *module,
                .pName = "main"
            },
            .layout = *computePipelineLayout
        };
        };

    // Scaffolding: In a live system, compile individual Slang/GLSL compute shaders for:
    // 1. "shaders/wavefront_raygen.spv"
    // 2. "shaders/wavefront_intersect.spv"
    // 3. "shaders/wavefront_shade.spv"
    // 4. "shaders/wavefront_tonemap.spv"
}

/*---------- COMMAND BUFFER: WAVEFRONT DISPATCH LOOP ----------*/

void SpectralWavefrontPt::recordWavefrontCommands(uint32_t imageIndex)
{
    auto& cmd = commandBuffers[frameIndex];
    cmd.begin({});

    WavefrontData::SceneParams params{
        .invView = glm::mat4(1.0f),
        .invProj = glm::mat4(1.0f),
        .frameCounter = frameIndex,
        .currentBounce = 0,
        .maxBounces = 8,
        .totalPixels = WIDTH * HEIGHT
    };

    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 0, *descriptorSets[0], nullptr);

    const uint32_t workgroupsX = (WIDTH + 15) / 16;
    const uint32_t workgroupsY = (HEIGHT + 15) / 16;

    // --- PHASE 1: RAY GENERATION ---
    // Generates camera rays, samples 4 spectral wavelengths per sample per pixel, populates RayQueue
    if (*pipelineRayGen) {
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelineRayGen);
        cmd.pushConstants<WavefrontData::SceneParams>(*computePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, params);
        cmd.dispatch(workgroupsX, workgroupsY, 1);
    }

    // Barrier: Ensure all rays are written to RayQueue before traversal
    vk::MemoryBarrier2 rayGenBarrier{
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
        if (*pipelineIntersect) {
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelineIntersect);
            cmd.dispatch(workgroupsX, workgroupsY, 1);
        }

        // Barrier: Wait for Intersect hits before shading
        vk::MemoryBarrier2 intersectBarrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead
        };
        depInfo.pMemoryBarriers = &intersectBarrier;
        cmd.pipelineBarrier2(depInfo);

        // 2b. Spectral Shade: Evaluates BSDF, Hero wavelength sampling, writes radiance to AccumulationImage
        if (*pipelineShade) {
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelineShade);
            cmd.dispatch(workgroupsX, workgroupsY, 1);
        }

        // Barrier: Wait for Shade writes to RayQueue before next bounce traversal
        vk::MemoryBarrier2 shadeBarrier{
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

    cmd.end();
}

/*---------- SWAPCHAIN, COMMANDS, AND RUNTIME BOILERPLATE ----------*/

void SpectralWavefrontPt::createSwapChain()
{
    auto caps = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    swapChainExtent = { WIDTH, HEIGHT };
    swapChainSurfaceFormat = { vk::Format::eB8G8R8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear };

    vk::SwapchainCreateInfoKHR createInfo{
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
    vk::ImageViewCreateInfo viewInfo{
        .viewType = vk::ImageViewType::e2D,
        .format = swapChainSurfaceFormat.format,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    };
    for (auto img : swapChainImages) {
        viewInfo.image = img;
        swapChainImageViews.emplace_back(device, viewInfo);
    }
}

void SpectralWavefrontPt::createCommandPool()
{
    vk::CommandPoolCreateInfo poolInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueIndex
    };
    commandPool = vk::raii::CommandPool(device, poolInfo);
}

void SpectralWavefrontPt::createCommandBuffers()
{
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = MAX_FRAMES_IN_FLIGHT
    };
    commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
}

void SpectralWavefrontPt::createSyncObjects()
{
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
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