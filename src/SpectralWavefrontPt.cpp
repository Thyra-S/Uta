//#include "SpectralWavefrontPt.h"
#include "ObjectLoading.h"
#define TINYEXR_USE_MINIZ 1
#define TINYEXR_USE_STB_ZLIB 0
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>

static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
    // 1. First priority: scRGB FP16 (True HDR)
    for (const auto& format : availableFormats)
    {
        if (format.format == vk::Format::eR16G16B16A16Sfloat &&
            format.colorSpace == vk::ColorSpaceKHR::eExtendedSrgbLinearEXT)
        {
            return format;
        }
    }

    // 2. Second priority: Standard SDR fallback
    for (const auto& format : availableFormats)
    {
        if (format.format == vk::Format::eB8G8R8A8Unorm &&
            format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
        {
            return format;
        }
        if (format.format == vk::Format::eR8G8B8A8Unorm &&
            format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
        {
            return format;
        }
    }

    return availableFormats[0];
}

void SpectralWavefrontPt::run(const std::string& modelPath, const std::string& materialPath)
{
    initWindow();
    initVulkan(modelPath, materialPath);
    mainLoop();
    cleanup();
}

void SpectralWavefrontPt::initWindow()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_RED_BITS, 16);
    glfwWindowHint(GLFW_GREEN_BITS, 16);
    glfwWindowHint(GLFW_BLUE_BITS, 16);
    glfwWindowHint(GLFW_ALPHA_BITS, 16);
    window = glfwCreateWindow(WIDTH, HEIGHT, "Uta Path Tracer", nullptr, nullptr);
}

void SpectralWavefrontPt::initVulkan(const std::string& modelPath, const std::string& materialPath)
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
    createResolvedImage();
    createStagingBuffer();
    createWavefrontQueues();
    createScene(modelPath, materialPath);
    createMaterialBuffer();
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
    extensions.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
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

    bool int64Supported = features2.template get <vk::PhysicalDeviceFeatures2>().features.shaderInt64;
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
            .shaderSampledImageArrayNonUniformIndexing = true,
            .runtimeDescriptorArray = true,
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
    vk::ImageCreateInfo imageInfo
    {
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

    vk::MemoryAllocateInfo allocInfo
    {
        .allocationSize = memReq.size,
        .memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
    };
    accumulationImageMemory = vk::raii::DeviceMemory(device, allocInfo);
    accumulationImage.bindMemory(*accumulationImageMemory, 0);

    vk::ImageViewCreateInfo viewInfo
    {
        .image = *accumulationImage,
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR32G32B32A32Sfloat,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    };
    accumulationImageView = vk::raii::ImageView(device, viewInfo);

    // One-time command buffer: transition eUndefined -> eGeneral and clear image to 0
    vk::CommandBufferAllocateInfo cmdAllocInfo
    {
        .commandPool = *commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };
    vk::raii::CommandBuffers cmds(device, cmdAllocInfo);
    auto& cmd = cmds[0];

    cmd.begin({ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

    vk::ImageMemoryBarrier2 initBarrier
    {
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

    vk::ImageMemoryBarrier2 toGeneralBarrier
    {
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
void SpectralWavefrontPt::createResolvedImage()
{
    vk::ImageCreateInfo imageInfo
    {
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .extent = { WIDTH, HEIGHT, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive
    };

    resolvedImage = vk::raii::Image(device, imageInfo);
    auto memReq = resolvedImage.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo
    {
        .allocationSize = memReq.size,
        .memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
    };
    resolvedImageMemory = vk::raii::DeviceMemory(device, allocInfo);
    resolvedImage.bindMemory(*resolvedImageMemory, 0);

    vk::ImageViewCreateInfo viewInfo
    {
        .image = *resolvedImage,
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    };
    resolvedImageView = vk::raii::ImageView(device, viewInfo);

    // Transition eUndefined -> eGeneral
    vk::CommandBufferAllocateInfo cmdAllocInfo
    {
        .commandPool = *commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };
    vk::raii::CommandBuffers cmds(device, cmdAllocInfo);
    auto& cmd = cmds[0];

    cmd.begin({ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

    vk::ImageMemoryBarrier2 barrier
    {
        .srcStageMask = vk::PipelineStageFlagBits2::eNone,
        .srcAccessMask = vk::AccessFlagBits2::eNone,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = *resolvedImage,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier });

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
void SpectralWavefrontPt::updateCameraMatrices()
{
    glm::mat4 view = glm::lookAt(camera.position, camera.target, camera.up);
    glm::mat4 proj = glm::perspective(
        glm::radians(camera.fovY),
        static_cast<float>(WIDTH) / static_cast<float>(HEIGHT),
        0.1f,
        1000.0f
    );

    invView = glm::inverse(view);
    invProj = glm::inverse(proj);
}

void SpectralWavefrontPt::createMaterialBuffer()
{
    if (materials.empty()) {
        materials.push_back(WavefrontData::GpuOpenPbrMaterial{});
    }

    const vk::DeviceSize bufferSize = materials.size() * sizeof(WavefrontData::GpuOpenPbrMaterial);
    createDeviceLocalBuffer(
        materials.data(),
        bufferSize,
        vk::BufferUsageFlagBits::eStorageBuffer,
        materialBuffer,
        materialMemory);

    materialBufferAddress = device.getBufferAddress({ .buffer = *materialBuffer });
}

void SpectralWavefrontPt::createSceneTextures(const std::vector<CpuTextureData>& cpuTextures)
{
    sceneTextures.clear();

    // 1. Create linear repeat sampler for texture lookups
    vk::SamplerCreateInfo samplerInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eRepeat,
        .addressModeV = vk::SamplerAddressMode::eRepeat,
        .addressModeW = vk::SamplerAddressMode::eRepeat,
        .mipLodBias = 0.0f,
        .anisotropyEnable = vk::False,
        .maxAnisotropy = 1.0f,
        .compareEnable = vk::False,
        .minLod = 0.0f,
        .maxLod = 0.0f,
        .borderColor = vk::BorderColor::eIntOpaqueBlack,
        .unnormalizedCoordinates = vk::False
    };
    linearSampler = vk::raii::Sampler(device, samplerInfo);

    // Ensure at least 1 fallback texture exists so Descriptor Set Binding 5 is never empty
    std::vector<CpuTextureData> fallbackList;
    const auto& texturesToUpload = cpuTextures.empty() ? [&]() -> const std::vector<CpuTextureData>&{
        fallbackList.push_back(CpuTextureData{
            .filepath = "fallback_white",
            .width = 1,
            .height = 1,
            .isSrgb = false,
            .pixels = { 255, 255, 255, 255 }
            });
        return fallbackList;
        }() : cpuTextures;

    sceneTextures.reserve(texturesToUpload.size());

    for (const auto& cpuTex : texturesToUpload)
    {
        vk::Format texFormat = cpuTex.isSrgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;
        vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(cpuTex.width) * cpuTex.height * 4;

        // Staging buffer
        vk::BufferCreateInfo stagingInfo{
            .size = imageSize,
            .usage = vk::BufferUsageFlagBits::eTransferSrc,
            .sharingMode = vk::SharingMode::eExclusive
        };
        vk::raii::Buffer tempStagingBuf(device, stagingInfo);
        auto stagingReq = tempStagingBuf.getMemoryRequirements();

        vk::MemoryAllocateInfo stagingAlloc{
            .allocationSize = stagingReq.size,
            .memoryTypeIndex = findMemoryType(
                stagingReq.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
        };
        vk::raii::DeviceMemory tempStagingMem(device, stagingAlloc);
        tempStagingBuf.bindMemory(*tempStagingMem, 0);

        void* mapped = tempStagingMem.mapMemory(0, imageSize);
        memcpy(mapped, cpuTex.pixels.data(), static_cast<size_t>(imageSize));
        tempStagingMem.unmapMemory();

        // Device-local Image
        WavefrontData::GpuTexture gpuTex{};
        vk::ImageCreateInfo imageInfo{
            .imageType = vk::ImageType::e2D,
            .format = texFormat,
            .extent = { cpuTex.width, cpuTex.height, 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            .sharingMode = vk::SharingMode::eExclusive
        };
        gpuTex.image = vk::raii::Image(device, imageInfo);
        auto imgReq = gpuTex.image.getMemoryRequirements();

        vk::MemoryAllocateInfo imgAlloc{
            .allocationSize = imgReq.size,
            .memoryTypeIndex = findMemoryType(imgReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
        };
        gpuTex.memory = vk::raii::DeviceMemory(device, imgAlloc);
        gpuTex.image.bindMemory(*gpuTex.memory, 0);

        // Transition Undefined -> TransferDst, Copy, Transition TransferDst -> ShaderReadOnlyOptimal
        vk::CommandBufferAllocateInfo cmdAlloc{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };
        vk::raii::CommandBuffers cmds(device, cmdAlloc);
        auto& cmd = cmds[0];
        cmd.begin({ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

        vk::ImageSubresourceRange subRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };

        vk::ImageMemoryBarrier2 toDst{
            .srcStageMask = vk::PipelineStageFlagBits2::eNone,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .image = *gpuTex.image,
            .subresourceRange = subRange
        };
        cmd.pipelineBarrier2(vk::DependencyInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toDst });

        vk::BufferImageCopy copyRegion{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 },
            .imageOffset = { 0, 0, 0 },
            .imageExtent = { cpuTex.width, cpuTex.height, 1 }
        };
        cmd.copyBufferToImage(*tempStagingBuf, *gpuTex.image, vk::ImageLayout::eTransferDstOptimal, { copyRegion });

        vk::ImageMemoryBarrier2 toShaderRead{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .image = *gpuTex.image,
            .subresourceRange = subRange
        };
        cmd.pipelineBarrier2(vk::DependencyInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toShaderRead });

        cmd.end();
        vk::SubmitInfo submitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*cmd };
        computeAndGraphicsQueue.submit(submitInfo, nullptr);
        computeAndGraphicsQueue.waitIdle();

        vk::ImageViewCreateInfo viewInfo{
            .image = *gpuTex.image,
            .viewType = vk::ImageViewType::e2D,
            .format = texFormat,
            .subresourceRange = subRange
        };
        gpuTex.view = vk::raii::ImageView(device, viewInfo);

        sceneTextures.push_back(std::move(gpuTex));
    }
}

void SpectralWavefrontPt::createDeviceLocalBuffer(
    const void* srcData,
    vk::DeviceSize size,
    vk::BufferUsageFlags usage,
    vk::raii::Buffer& dstBuffer,
    vk::raii::DeviceMemory& dstMemory)
{
    // 1. Create temporary CPU-visible staging buffer
    vk::BufferCreateInfo stagingInfo{
        .size = size,
        .usage = vk::BufferUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive
    };
    vk::raii::Buffer tempStagingBuffer(device, stagingInfo);
    auto stagingMemReq = tempStagingBuffer.getMemoryRequirements();

    vk::MemoryAllocateInfo stagingAllocInfo{
        .allocationSize = stagingMemReq.size,
        .memoryTypeIndex = findMemoryType(
            stagingMemReq.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
    };
    vk::raii::DeviceMemory tempStagingMemory(device, stagingAllocInfo);
    tempStagingBuffer.bindMemory(*tempStagingMemory, 0);

    void* mapped = tempStagingMemory.mapMemory(0, size);
    memcpy(mapped, srcData, size);
    tempStagingMemory.unmapMemory();

    // 2. Create permanent VRAM (eDeviceLocal) buffer with Device Address support
    vk::BufferCreateInfo dstInfo{
        .size = size,
        .usage = usage | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        .sharingMode = vk::SharingMode::eExclusive
    };
    dstBuffer = vk::raii::Buffer(device, dstInfo);
    auto dstMemReq = dstBuffer.getMemoryRequirements();

    vk::MemoryAllocateFlagsInfo flagsInfo{ .flags = vk::MemoryAllocateFlagBits::eDeviceAddress };
    vk::MemoryAllocateInfo dstAllocInfo{
        .pNext = &flagsInfo,
        .allocationSize = dstMemReq.size,
        .memoryTypeIndex = findMemoryType(
            dstMemReq.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eDeviceLocal)
    };
    dstMemory = vk::raii::DeviceMemory(device, dstAllocInfo);
    dstBuffer.bindMemory(*dstMemory, 0);

    // 3. Copy staging -> VRAM via one-time command buffer
    vk::CommandBufferAllocateInfo cmdAllocInfo{
        .commandPool = *commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };
    vk::raii::CommandBuffers cmds(device, cmdAllocInfo);
    auto& cmd = cmds[0];

    cmd.begin({ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

    vk::BufferCopy copyRegion{ .srcOffset = 0, .dstOffset = 0, .size = size };
    cmd.copyBuffer(*tempStagingBuffer, *dstBuffer, { copyRegion });

    // Memory barrier ensuring transfer finishes before AS build or Compute Shader reads
    vk::MemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR | vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eShaderRead
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{ .memoryBarrierCount = 1, .pMemoryBarriers = &barrier });

    cmd.end();

    vk::SubmitInfo submitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*cmd };
    computeAndGraphicsQueue.submit(submitInfo, nullptr);
    computeAndGraphicsQueue.waitIdle();
    // tempStagingBuffer & tempStagingMemory automatically clean up on scope exit
}

void SpectralWavefrontPt::createScene(const std::string& modelPath, const std::string& materialPath)
{
    meshes.clear();
    instances.clear();
    materials.clear();

    std::vector<CpuTextureData> cpuTextures;
    std::unordered_map<std::string, uint32_t> textureCache;

    // 3. Load USD scene geometry, camera, and any USD-authored materials/textures
    //    (Meshes without a USD material binding default to materialId = 0, i.e., primaryMtlx)
    loadUsdScene(modelPath, meshes, instances, materials, cpuTextures, camera);

    // 5. Upload all discovered textures to GPU
    createSceneTextures(cpuTextures);

	createEmissiveTriangleBuffer();
    updateCameraMatrices();
}

void SpectralWavefrontPt::createEmissiveTriangleBuffer()
{
    std::vector<WavefrontData::GpuEmissiveTriangle> emissiveTriangles;
    for (const auto& inst : instances)
    {
        const auto& material = materials[inst.materialId];
        if (material.emissionAndLuminance.w <= 0.0f)
            continue; // Skip non-emissive instances

        const auto& mesh = meshes[inst.meshIndex];
        for (size_t i = 0; i < mesh.indices.size(); i += 3)
        {
            uint32_t idx0 = mesh.indices[i];
            uint32_t idx1 = mesh.indices[i + 1];
            uint32_t idx2 = mesh.indices[i + 2];
            const auto& v0 = glm::vec3(inst.transform * glm::vec4(mesh.vertices[idx0], 1.0f));
            const auto& v1 = glm::vec3(inst.transform * glm::vec4(mesh.vertices[idx1], 1.0f));
            const auto& v2 = glm::vec3(inst.transform * glm::vec4(mesh.vertices[idx2], 1.0f));
            // Compute triangle area and normal
            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;
            float area = 0.5f * glm::length(glm::cross(edge1, edge2));

            WavefrontData::GpuEmissiveTriangle emissiveTri
            {
                .v0            = v0,
                .area          = area,
                .v1            = v1,
                .luminance     = material.emissionAndLuminance.w,
                .v2            = v2,
                .materialId    = inst.materialId,
                .emissionColor = {glm::vec3(material.emissionAndLuminance) * material.emissionAndLuminance.w, 0.0f} // Pad to vec4
            };
            emissiveTriangles.push_back(emissiveTri);
        }
    }

    emissiveTriangleCount = static_cast<uint32_t>(emissiveTriangles.size());
    if (emissiveTriangles.empty())
    {
        WavefrontData::GpuEmissiveTriangle emissiveTri
        {
            .v0            = {0.0f, 0.0f, 0.0f},
            .area          = 0.0f,
            .v1            = {0.0f, 0.0f, 0.0f},
            .luminance     = 0.0f,
            .v2            = {0.0f, 0.0f, 0.0f},
            .materialId    = 0,
            .emissionColor = {0.0f, 0.0f, 0.0f, 0.0f} // Pad to vec4
        };
        emissiveTriangles.push_back(emissiveTri);
    }

    const vk::DeviceSize bufferSize = emissiveTriangles.size() * sizeof(WavefrontData::GpuEmissiveTriangle);
    createDeviceLocalBuffer(
        emissiveTriangles.data(),
        bufferSize,
        vk::BufferUsageFlagBits::eStorageBuffer,
        emissiveTriangleBuffer,
        emissiveTriangleMemory);
    emissiveTriangleBufferAddress = device.getBufferAddress({ .buffer = *emissiveTriangleBuffer });
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
    const vk::BufferUsageFlags geomUsage =
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
        vk::BufferUsageFlagBits::eStorageBuffer;

    createDeviceLocalBuffer(
        mesh.vertices.data(),
        mesh.vertices.size() * sizeof(glm::vec3),
        geomUsage,
        mesh.vertexBuffer,
        mesh.vertexBufferMemory);

    createDeviceLocalBuffer(
        mesh.attributes.data(),
        mesh.attributes.size() * sizeof(VertexAttributes),
        vk::BufferUsageFlagBits::eStorageBuffer,
        mesh.attributeBuffer,
        mesh.attributeBufferMemory);

    createDeviceLocalBuffer(
        mesh.indices.data(),
        mesh.indices.size() * sizeof(uint32_t),
        geomUsage,
        mesh.indexBuffer,
        mesh.indexBufferMemory);

    vk::DeviceAddress vertexAddress = device.getBufferAddress({ .buffer = *mesh.vertexBuffer });
    vk::DeviceAddress indexAddress = device.getBufferAddress({ .buffer = *mesh.indexBuffer });

    vk::AccelerationStructureGeometryTrianglesDataKHR triangles{
        .vertexFormat = vk::Format::eR32G32B32Sfloat,
        .vertexData = {.deviceAddress = vertexAddress },
        .vertexStride = sizeof(glm::vec3),
        .maxVertex = static_cast<uint32_t>(mesh.vertices.size() - 1), // Note: Vulkan spec expects max index (size - 1)
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

    // Allocate BLAS storage buffer in DeviceLocal VRAM!
    vk::BufferCreateInfo blasBufInfo{
        .size = buildSizes.accelerationStructureSize,
        .usage = vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        .sharingMode = vk::SharingMode::eExclusive
    };
    mesh.blasBuffer = vk::raii::Buffer(device, blasBufInfo);
    auto blasMemReq = mesh.blasBuffer.getMemoryRequirements();

    vk::MemoryAllocateFlagsInfo flagsInfo{ .flags = vk::MemoryAllocateFlagBits::eDeviceAddress };
    vk::MemoryAllocateInfo blasAllocInfo{
        .pNext = &flagsInfo,
        .allocationSize = blasMemReq.size,
        .memoryTypeIndex = findMemoryType(blasMemReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
    };
    mesh.blasMemory = vk::raii::DeviceMemory(device, blasAllocInfo);
    mesh.blasBuffer.bindMemory(*mesh.blasMemory, 0);

    vk::AccelerationStructureCreateInfoKHR createInfo{
        .buffer = *mesh.blasBuffer,
        .size = buildSizes.accelerationStructureSize,
        .type = vk::AccelerationStructureTypeKHR::eBottomLevel
    };
    mesh.blas = vk::raii::AccelerationStructureKHR(device, createInfo);

    buildAccelerationStructureOnGPU(buildInfo, *mesh.blas, buildSizes.buildScratchSize, primitiveCount);

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
    std::vector<WavefrontData::GpuMeshDesc> descriptors;
    descriptors.reserve(instances.size());

    for (const auto& inst : instances) {
        const auto& mesh = meshes[inst.meshIndex];
        descriptors.push_back(WavefrontData::GpuMeshDesc{
            .positionBufferAddress = device.getBufferAddress({.buffer = *mesh.vertexBuffer }),
            .attributtesBufferAddress = device.getBufferAddress({.buffer = *mesh.attributeBuffer }),
            .indexBufferAddress = device.getBufferAddress({.buffer = *mesh.indexBuffer }),
            .materialId = inst.materialId,
            .padding = 0
            });
    }

    vk::DeviceSize bufferSize = sizeof(WavefrontData::GpuMeshDesc) * descriptors.size();
    createDeviceLocalBuffer(
        descriptors.data(),
        bufferSize,
        vk::BufferUsageFlagBits::eStorageBuffer,
        meshDescBuffer,
        meshDescMemory);

    meshDescAddress = device.getBufferAddress({ .buffer = *meshDescBuffer });
}

/*---------- DESCRIPTORS & COMPUTE PIPELINES ----------*/

void SpectralWavefrontPt::createDescriptorSetLayout()
{
    // Binding 0: Top-Level Acceleration Structure (TLAS) accessed via Ray Query
    // Binding 1: Offscreen HDR Accumulation Image (RW Storage Image)
    // Binding 2: Ray Queue (SSBO)
    // Binding 3: Hit Queue (SSBO)
    // Binding 4: Resolved SDR Image (RW Storage Image)
	// Binding 5: Scene Textures (Sampled Image Array) - optional, may be empty if no textures are used
    std::array<vk::DescriptorSetLayoutBinding, 6> bindings{ {
        {
            .binding = 0,\
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
        },
        {
            .binding = 4,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute
        },
        {
            .binding = 5,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = static_cast<uint32_t>(sceneTextures.size()),
            .stageFlags = vk::ShaderStageFlagBits::eCompute
        }
    } };

    vk::DescriptorSetLayoutCreateInfo layoutInfo
    {
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()
    };
    descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

    vk::PushConstantRange pushConstantRange
    {
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(WavefrontData::SceneParams)
    };

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo
    {
        .setLayoutCount = 1,
        .pSetLayouts = &*descriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange
    };
    computePipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);
}

void SpectralWavefrontPt::createDescriptorPoolAndSets()
{
    uint32_t texCount = static_cast<uint32_t>(sceneTextures.size());
    std::array<vk::DescriptorPoolSize, 4> poolSizes
    { {
        { vk::DescriptorType::eAccelerationStructureKHR, 1 },
        { vk::DescriptorType::eStorageImage, 2 },
        { vk::DescriptorType::eStorageBuffer, 2 },
        { vk::DescriptorType::eCombinedImageSampler, texCount }
    } };

    vk::DescriptorPoolCreateInfo poolInfo
    {
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 1,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };
    descriptorPool = vk::raii::DescriptorPool(device, poolInfo);

    vk::DescriptorSetAllocateInfo allocInfo
    {
        .descriptorPool = *descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &*descriptorSetLayout
    };
    descriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    // Acceleration structure write descriptor
    vk::WriteDescriptorSetAccelerationStructureKHR asInfo
    {
        .accelerationStructureCount = 1,
        .pAccelerationStructures = &*tlas
    };

    // Bind Buffers and Images to Descriptor Sets
    vk::DescriptorImageInfo accumImageInfo
    {
        .imageView = *accumulationImageView,
        .imageLayout = vk::ImageLayout::eGeneral
    };
    vk::DescriptorImageInfo resolvedImageInfo
    {
        .imageView = *resolvedImageView,
        .imageLayout = vk::ImageLayout::eGeneral
    };
    vk::DescriptorBufferInfo rayBufferInfo
    {
        .buffer = *rayQueueBuffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE
    };
    vk::DescriptorBufferInfo hitBufferInfo
    {
        .buffer = *hitQueueBuffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE
    };

    std::vector<vk::DescriptorImageInfo> texImageInfos;
    texImageInfos.reserve(sceneTextures.size());
    for (const auto& tex : sceneTextures) {
        texImageInfos.push_back(vk::DescriptorImageInfo{
            .sampler = *linearSampler,
            .imageView = *tex.view,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
            });
    }

    std::array<vk::WriteDescriptorSet, 6> descriptorWrites
    { {
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
        },
        {
            .dstSet = *descriptorSets[0],
            .dstBinding = 4,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &resolvedImageInfo
        },
        {
            .dstSet = *descriptorSets[0],
            .dstBinding = 5,
            .dstArrayElement = 0,
            .descriptorCount = static_cast<uint32_t>(texImageInfos.size()),
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = texImageInfos.data()
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

    bool isScRgb = (swapChainSurfaceFormat.format == vk::Format::eR16G16B16A16Sfloat &&
        swapChainSurfaceFormat.colorSpace == vk::ColorSpaceKHR::eExtendedSrgbLinearEXT);

    WavefrontData::SceneParams params{
        .invView = invView,
        .invProj = invProj,
        .vertexBufferAddress = meshDescAddress,
        .materialBufferAddress = materialBufferAddress,
        .shiftOffset = glm::vec2(0.0f),
        .tiltAxis = glm::vec2(0.0f, 1.0f),
        .tiltAngle = 0.0f,
        .focalDistance = 5.0f,
        .apertureRadius = 0.0f,
        .width = WIDTH,
        .height = HEIGHT,
        .frameCounter = totalSamplesAccumulated,
        .currentBounce = 0,
        .maxBounces = 16,
        .totalPixels = WIDTH * HEIGHT,
        .pad0 = isScRgb ? 1u : 0u,
        .lightBufferAddress = emissiveTriangleBufferAddress,
        .lightCount = emissiveTriangleCount,
        .pad1 = 0,
    };

    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 0, *descriptorSets[0], nullptr);

    const uint32_t workgroupsX = (WIDTH + 7) / 8;
    const uint32_t workgroupsY = (HEIGHT + 7) / 8;
    const uint32_t totalPixels = WIDTH * HEIGHT;
    const uint32_t wavefrontWorkgroups = (totalPixels + 63) / 64;

    // --- PHASE 1: RAY GENERATION ---
    // Generates camera rays, samples 4 spectral wavelengths per sample per pixel, populates RayQueue
    
    uint32_t samplesThisFrame = SAMPLES_PER_FRAME;
    if (!autoExportDone && totalSamplesAccumulated < targetSamples)
    {
        samplesThisFrame = std::min(SAMPLES_PER_FRAME, targetSamples - totalSamplesAccumulated);
    }

    for (uint32_t s = 0; s < samplesThisFrame; s++)
    {
        params.frameCounter = totalSamplesAccumulated + s;
        params.currentBounce = 0;

        // --- PHASE 1: RAY GENERATION ---
        if (*pipelineRayGen)
        {
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelineRayGen);
            cmd.pushConstants<WavefrontData::SceneParams>(*computePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, params);
            cmd.dispatch(workgroupsX, workgroupsY, 1);
        }

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

            if (*pipelineIntersect)
            {
                cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelineIntersect);
                cmd.dispatch(workgroupsX, workgroupsY, 1);
            }

            vk::MemoryBarrier2 intersectBarrier{
                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead
            };
            depInfo.pMemoryBarriers = &intersectBarrier;
            cmd.pipelineBarrier2(depInfo);

            if (*pipelineShade)
            {
                cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelineShade);
                cmd.dispatch(workgroupsX, workgroupsY, 1);
            }

            // Wait for Shade writes to RayQueue AND AccumulationImage before next bounce OR next sample's RayGen
            vk::MemoryBarrier2 shadeBarrier{
                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite
            };
            depInfo.pMemoryBarriers = &shadeBarrier;
            cmd.pipelineBarrier2(depInfo);
        }
    }

    // --- PHASE 4: COMPOSITE / TONEMAP & PRESENTATION ---
    // In a production engine, blit or run a final tone-mapping compute pass converting XYZ to sRGB onto swapChainImages[imageIndex]
    vk::ImageMemoryBarrier2 accumBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = *accumulationImage,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &accumBarrier });

    if (*pipelineTonemap)
    {
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelineTonemap);
        cmd.pushConstants<WavefrontData::SceneParams>(*computePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, params);
        cmd.dispatch(workgroupsX, workgroupsY, 1);
    }

    // --- PHASE 5: PRESENTATION BLIT (resolvedImage -> swapchain) ---
    vk::ImageMemoryBarrier2 resolvedToSrcBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eTransferSrcOptimal,
        .image = *resolvedImage,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    };

    vk::ImageMemoryBarrier2 swapchainToDstBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eNone,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .image = swapChainImages[imageIndex],
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    };

    std::array<vk::ImageMemoryBarrier2, 2> preBlitBarriers = { resolvedToSrcBarrier, swapchainToDstBarrier };
    cmd.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<uint32_t>(preBlitBarriers.size()),
        .pImageMemoryBarriers = preBlitBarriers.data()
        });

    vk::ImageBlit blitRegion{
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
        *resolvedImage, vk::ImageLayout::eTransferSrcOptimal,
        swapChainImages[imageIndex], vk::ImageLayout::eTransferDstOptimal,
        { blitRegion }, vk::Filter::eNearest
    );

    // Restore resolvedImage back to eGeneral (for next frame or screenshot read)
    vk::ImageMemoryBarrier2 restoreResolvedBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = *resolvedImage,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    };

    vk::ImageMemoryBarrier2 swapchainToPresentBarrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eNone,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::ePresentSrcKHR,
        .image = swapChainImages[imageIndex],
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    };

    std::array<vk::ImageMemoryBarrier2, 2> postBlitBarriers = { restoreResolvedBarrier, swapchainToPresentBarrier };
    cmd.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<uint32_t>(postBlitBarriers.size()),
        .pImageMemoryBarriers = postBlitBarriers.data()
        });

    cmd.end();
}

/*---------- SWAPCHAIN, COMMANDS, AND RUNTIME BOILERPLATE ----------*/

void SpectralWavefrontPt::createSwapChain()
{
    auto caps = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    auto availableFormats = physicalDevice.getSurfaceFormatsKHR(*surface);

    if (availableFormats.empty()) {
        throw std::runtime_error("No available surface formats found for swapchain!");
    }

    swapChainExtent = { WIDTH, HEIGHT };
    swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);


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
        .presentMode = vk::PresentModeKHR::eImmediate,
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
    presentCompleteSemaphores.clear();
    renderFinishedSemaphores.clear();
    inFlightFences.clear();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});
        inFlightFences.emplace_back(device, vk::FenceCreateInfo{ .flags = vk::FenceCreateFlagBits::eSignaled });
    }
    for (size_t i = 0; i < swapChainImages.size(); i++) {
        renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});
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
    vk::SubmitInfo submitInfo
    {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*presentCompleteSemaphores[frameIndex],
        .pWaitDstStageMask = &waitMask,
        .commandBufferCount = 1,
        .pCommandBuffers = &*commandBuffers[frameIndex],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]
    };

    computeAndGraphicsQueue.submit(submitInfo, *inFlightFences[frameIndex]);

    vk::PresentInfoKHR presentInfo
    {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
        .swapchainCount = 1,
        .pSwapchains = &*swapChain,
        .pImageIndices = &imageIndex
    };

    result = computeAndGraphicsQueue.presentKHR(presentInfo);

    uint32_t samplesThisFrame = SAMPLES_PER_FRAME;
    if (!autoExportDone && totalSamplesAccumulated < targetSamples)
    {
        samplesThisFrame = std::min(SAMPLES_PER_FRAME, targetSamples - totalSamplesAccumulated);
    }

    totalSamplesAccumulated += samplesThisFrame;
    frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;

    if (!autoExportDone && totalSamplesAccumulated >= targetSamples)
    {
        saveScreenshotAndLog("auto_benchmark");
        autoExportDone = true;
    }
}

void SpectralWavefrontPt::mainLoop()
{
    renderStartTime = std::chrono::high_resolution_clock::now();

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
    createSyncObjects();
}

void SpectralWavefrontPt::cleanup()
{
    device.waitIdle();
    cleanupSwapChain();
    stagingBuffer = nullptr;
    stagingBufferMemory = nullptr;
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

/*---------- BENCHMARK & OUTPUT SETTINGS ----------*/

void SpectralWavefrontPt::createStagingBuffer()
{
    // 4 channels x 32 bit float
    vk::DeviceSize bufferSize = static_cast<vk::DeviceSize>(WIDTH) * HEIGHT * 4 * sizeof(float);

    vk::BufferCreateInfo bufferInfo
    {
        .size = bufferSize,
        .usage = vk::BufferUsageFlagBits::eTransferDst,
        .sharingMode = vk::SharingMode::eExclusive
    };
    stagingBuffer = vk::raii::Buffer(device, bufferInfo);

    auto memReqs = stagingBuffer.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo
    {
        .allocationSize = memReqs.size,
        .memoryTypeIndex = findMemoryType(
            memReqs.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        )
    };
    stagingBufferMemory = vk::raii::DeviceMemory(device, allocInfo);
    stagingBuffer.bindMemory(*stagingBufferMemory, 0);
}

void SpectralWavefrontPt::saveScreenshotAndLog(const std::string& prefix)
{
    device.waitIdle();

    auto currentTime = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(currentTime - renderStartTime).count();
    double elapsedSec = elapsedMs / 1000.0;

    // 1. One-time command buffer to copy accumulationImage (RGBA32F) to the CPU staging buffer
    vk::CommandBufferAllocateInfo cmdAllocInfo
    {
        .commandPool = *commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };
    vk::raii::CommandBuffers tempCmds(device, cmdAllocInfo);
    auto& cmd = tempCmds[0];

    cmd.begin({ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

    // Transition accumulationImage from eGeneral -> eTransferSrcOptimal
    vk::ImageMemoryBarrier2 toSrcBarrier
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
    cmd.pipelineBarrier2(vk::DependencyInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toSrcBarrier });

    vk::BufferImageCopy copyRegion
    {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 },
        .imageOffset = { 0, 0, 0 },
        .imageExtent = { WIDTH, HEIGHT, 1 }
    };
    cmd.copyImageToBuffer(*accumulationImage, vk::ImageLayout::eTransferSrcOptimal, *stagingBuffer, copyRegion);

    // Transition accumulationImage back to eGeneral for subsequent frames
    vk::ImageMemoryBarrier2 toGenBarrier
    {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
        .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = *accumulationImage,
        .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toGenBarrier });

    cmd.end();

    vk::SubmitInfo submitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*cmd };
    computeAndGraphicsQueue.submit(submitInfo, nullptr);
    computeAndGraphicsQueue.waitIdle();

    // 2. Map 32-bit float pixels (interleaved RGBA32F)
    const vk::DeviceSize bufferSize = static_cast<vk::DeviceSize>(WIDTH) * HEIGHT * 4 * sizeof(float);
    auto* mappedData = static_cast<const float*>(stagingBufferMemory.mapMemory(0, bufferSize));

    // 3. Separate channels into planar 32-bit float buffers and divide by sample count
    const size_t pixelCount = static_cast<size_t>(WIDTH) * HEIGHT;
    std::vector<float> rChannel(pixelCount);
    std::vector<float> gChannel(pixelCount);
    std::vector<float> bChannel(pixelCount);

    const float invSamples = 1.0f / static_cast<float>(std::max(1u, totalSamplesAccumulated));

    for (size_t i = 0; i < pixelCount; ++i)
    {
        rChannel[i] = mappedData[i * 4 + 0] * invSamples;
        gChannel[i] = mappedData[i * 4 + 1] * invSamples;
        bChannel[i] = mappedData[i * 4 + 2] * invSamples;
    }
    stagingBufferMemory.unmapMemory();

    auto now = std::chrono::system_clock::now();
    std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    std::tm localTm{};
#if defined(_WIN32)
    localtime_s(&localTm, &nowTime);
#else
    localtime_r(&nowTime, &localTm);
#endif

    std::ostringstream timestampStream;
    timestampStream << std::put_time(&localTm, "%Y%m%d_%H%M%S");
    std::string timestamp = timestampStream.str();

    // Ensure output directories exist so SaveEXRImageToFile doesn't fail
    std::filesystem::create_directories("renders");
    std::filesystem::create_directories("logs");

    std::string baseName = prefix + "_" + std::to_string(totalSamplesAccumulated) + "spp_" + timestamp;
    std::string exrFilename = "renders/" + baseName + ".exr";
    std::string txtFilename = "logs/" + baseName + "_benchmark.txt";

    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;

    const unsigned char* imagePtr[3] = {
        reinterpret_cast<const unsigned char*>(bChannel.data()),
        reinterpret_cast<const unsigned char*>(gChannel.data()),
        reinterpret_cast<const unsigned char*>(rChannel.data())
    };

    image.images = const_cast<unsigned char**>(imagePtr);
    image.width = static_cast<int>(WIDTH);
    image.height = static_cast<int>(HEIGHT);

    header.num_channels = 3;
    header.channels = static_cast<EXRChannelInfo*>(malloc(sizeof(EXRChannelInfo) * header.num_channels));
    strncpy(header.channels[0].name, "B", 255);
    strncpy(header.channels[1].name, "G", 255);
    strncpy(header.channels[2].name, "R", 255);

    header.pixel_types = static_cast<int*>(malloc(sizeof(int) * header.num_channels));
    header.requested_pixel_types = static_cast<int*>(malloc(sizeof(int) * header.num_channels));
    for (int i = 0; i < header.num_channels; ++i)
    {
        // Input arrays (rChannel, gChannel, bChannel) are 32-bit floats
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
        // Request 16-bit half-float output on disk (or TINYEXR_PIXELTYPE_FLOAT for full FP32 EXR)
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF;
    }

    header.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP;
    const char* err = nullptr;
    int ret = SaveEXRImageToFile(&image, &header, exrFilename.c_str(), &err);
    if (ret != TINYEXR_SUCCESS)
    {
        std::cerr << "[EXR Error] " << (err ? err : "Unknown") << std::endl;
        FreeEXRErrorMessage(err);
    }

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);

    // 5. Write benchmark timing file
    std::ofstream txtFile(txtFilename);
    txtFile << "================ Rendering Benchmark ================\n";
    txtFile << "Resolution:        " << WIDTH << " x " << HEIGHT << "\n";
    txtFile << "Accumulated SPP:   " << totalSamplesAccumulated << "\n";
    txtFile << "Total Time:        " << std::fixed << std::setprecision(3) << elapsedSec << " s (" << elapsedMs << " ms)\n";
    txtFile << "Average per Sample:" << (elapsedMs / std::max(1u, totalSamplesAccumulated)) << " ms/spp\n";
    txtFile << "Output Image:      " << exrFilename << "\n";
    txtFile << "====================================================\n";
    txtFile.close();

    std::cout << "[Export] Saved " << exrFilename << " and " << txtFilename << std::endl;
}