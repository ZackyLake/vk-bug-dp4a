#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef VK_COMPUTE_SHADER_PATH
#error VK_COMPUTE_SHADER_PATH must point to the compiled shader
#endif

namespace {

constexpr std::uint32_t kPackedInput = 0x04030201u;
constexpr std::size_t kInvocationCount = 16;
constexpr std::array<std::uint32_t, 2> kPipelineBlockCounts = {4, 32};
constexpr std::uint32_t kMinimumSubgroupSize = 16;
constexpr VkDeviceSize kInputElementStride = 8;
constexpr VkDeviceSize kInputBufferSize = kInvocationCount * kInputElementStride;
constexpr VkDeviceSize kResultBufferSize = kInvocationCount * sizeof(std::uint8_t);

struct InputElement {
    std::uint32_t data;
    std::uint8_t value;
    std::array<std::uint8_t, 3> padding{};
};

static_assert(sizeof(InputElement) == kInputElementStride);

struct ExpectedResults {
    std::array<std::uint8_t, kInvocationCount> values;
};

struct CommandLineOptions {
    std::optional<std::uint32_t> deviceIndex;
    std::optional<std::uint32_t> simdSize;
};

struct SubgroupInfo {
    std::uint32_t subgroupSize = 0;
    std::uint32_t minSubgroupSize = 0;
    std::uint32_t maxSubgroupSize = 0;
    VkShaderStageFlags requiredSubgroupSizeStages = 0;
    bool subgroupSizeControlAvailable = false;
    bool subgroupSizeControlFeature = false;

    bool supportsRequiredSubgroupSize() const {
        return subgroupSizeControlAvailable && subgroupSizeControlFeature &&
               (requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
    }
};

struct SubgroupSelection {
    std::uint32_t size = 0;
    bool useRequiredSubgroupSize = false;
};

std::array<InputElement, kInvocationCount> makeInputData() {
    std::array<InputElement, kInvocationCount> input{};
    for (std::size_t index = 0; index < kInvocationCount; ++index) {
        input[index].data = kPackedInput + static_cast<std::uint32_t>(index);
        input[index].value = static_cast<std::uint8_t>(index);
    }
    return input;
}

ExpectedResults makeExpectedResults(const std::array<InputElement, kInvocationCount>& input) {
    ExpectedResults expected{};
    for (std::size_t index = 0; index < kInvocationCount; ++index) {
        expected.values[index] = static_cast<std::uint8_t>(input[index].data);
    }
    return expected;
}

void checkVk(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with VkResult " +
                                 std::to_string(static_cast<int>(result)));
    }
}

void printStatisticValue(const VkPipelineExecutableStatisticKHR& statistic) {
    switch (statistic.format) {
    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
        std::cout << (statistic.value.b32 == VK_TRUE ? "true" : "false");
        break;
    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
        std::cout << statistic.value.i64;
        break;
    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
        std::cout << statistic.value.u64;
        break;
    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
        std::cout << statistic.value.f64;
        break;
    default:
        std::cout << "<unknown>";
        break;
    }
}

CommandLineOptions parseCommandLine(int argc, char* argv[]) {
    CommandLineOptions options;
    for (int argumentIndex = 1; argumentIndex < argc; ++argumentIndex) {
        const std::string argument = argv[argumentIndex];
        if ((argument != "-i" && argument != "--simd") ||
            argumentIndex + 1 >= argc) {
            throw std::runtime_error(
                "Usage: vulkan_compute_demo [-i device_index] [--simd subgroup_size]");
        }

        const std::string value = argv[++argumentIndex];
        std::uint32_t parsedIndex = 0;
        const auto parseResult = std::from_chars(
            value.data(), value.data() + value.size(), parsedIndex);
        if (parseResult.ec != std::errc{} || parseResult.ptr != value.data() + value.size()) {
            throw std::runtime_error(
                argument == "-i" ? "Device index must be a non-negative integer"
                                  : "SIMD size must be a non-negative integer");
        }
        if (argument == "-i") {
            if (options.deviceIndex.has_value()) {
                throw std::runtime_error("Device index was specified more than once");
            }
            options.deviceIndex = parsedIndex;
        } else {
            if (options.simdSize.has_value()) {
                throw std::runtime_error("SIMD size was specified more than once");
            }
            options.simdSize = parsedIndex;
        }
    }
    return options;
}

std::vector<VkPhysicalDevice> getPhysicalDevices(VkInstance instance) {
    std::uint32_t deviceCount = 0;
    checkVk(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr),
            "vkEnumeratePhysicalDevices");
    if (deviceCount == 0) {
        throw std::runtime_error("No Vulkan physical devices found");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    checkVk(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()),
            "vkEnumeratePhysicalDevices");
    devices.resize(deviceCount);
    return devices;
}

bool hasPhysicalDeviceExtension(VkPhysicalDevice physicalDevice, const char* requiredExtension) {
    std::uint32_t extensionCount = 0;
    checkVk(vkEnumerateDeviceExtensionProperties(
                physicalDevice, nullptr, &extensionCount, nullptr),
            "vkEnumerateDeviceExtensionProperties");
    std::vector<VkExtensionProperties> extensions(extensionCount);
    checkVk(vkEnumerateDeviceExtensionProperties(
                physicalDevice, nullptr, &extensionCount, extensions.data()),
            "vkEnumerateDeviceExtensionProperties");
    for (const auto& extension : extensions) {
        if (std::string(extension.extensionName) == requiredExtension) {
            return true;
        }
    }
    return false;
}

SubgroupInfo getSubgroupInfo(VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceProperties deviceProperties{};
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    const bool subgroupSizeControlAvailable =
        deviceProperties.apiVersion >= VK_API_VERSION_1_3 ||
        hasPhysicalDeviceExtension(physicalDevice, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);

    VkPhysicalDeviceSubgroupProperties subgroupProperties{};
    subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceSubgroupSizeControlProperties sizeControlProperties{};
    sizeControlProperties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
    if (subgroupSizeControlAvailable) {
        subgroupProperties.pNext = &sizeControlProperties;
    }
    VkPhysicalDeviceProperties2 properties2{};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &subgroupProperties;
    vkGetPhysicalDeviceProperties2(physicalDevice, &properties2);

    VkPhysicalDeviceSubgroupSizeControlFeatures sizeControlFeatures{};
    sizeControlFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    if (subgroupSizeControlAvailable) {
        features2.pNext = &sizeControlFeatures;
    }
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

    SubgroupInfo info{};
    info.subgroupSize = subgroupProperties.subgroupSize;
    info.subgroupSizeControlAvailable = subgroupSizeControlAvailable;
    if (subgroupSizeControlAvailable) {
        info.minSubgroupSize = sizeControlProperties.minSubgroupSize;
        info.maxSubgroupSize = sizeControlProperties.maxSubgroupSize;
        info.requiredSubgroupSizeStages = sizeControlProperties.requiredSubgroupSizeStages;
        info.subgroupSizeControlFeature = sizeControlFeatures.subgroupSizeControl == VK_TRUE;
    }
    return info;
}

SubgroupSelection selectSubgroupSize(
    const SubgroupInfo& info,
    const std::optional<std::uint32_t>& requestedSimdSize) {
    if (!info.supportsRequiredSubgroupSize()) {
        if (info.subgroupSize == 0) {
            throw std::runtime_error(
                "Selected device does not report a native subgroup size");
        }
        return {info.subgroupSize, false};
    }

    const std::uint32_t selectedSize = requestedSimdSize.value_or(
        std::max(kMinimumSubgroupSize, info.minSubgroupSize));
    if (selectedSize < info.minSubgroupSize || selectedSize > info.maxSubgroupSize) {
        throw std::runtime_error(
            "Requested SIMD size " + std::to_string(selectedSize) +
            " is outside the selected device range [" +
            std::to_string(info.minSubgroupSize) + ", " +
            std::to_string(info.maxSubgroupSize) + "]");
    }
    return {selectedSize, true};
}

std::uint32_t chooseDeviceIndex(
    const std::vector<VkPhysicalDevice>& devices,
    const CommandLineOptions& options) {
    if (options.deviceIndex.has_value()) {
        if (*options.deviceIndex >= devices.size()) {
            throw std::runtime_error("Device index is out of range");
        }
        return *options.deviceIndex;
    }

    for (std::size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(devices[deviceIndex], &properties);
        const auto subgroupInfo = getSubgroupInfo(devices[deviceIndex]);
        std::cout << deviceIndex << ": " << properties.deviceName
                  << " (";
        if (!subgroupInfo.supportsRequiredSubgroupSize()) {
            std::cout << "subgroup(default)=" << subgroupInfo.subgroupSize;
        } else {
            const std::uint32_t selectedSize = options.simdSize.value_or(
                std::max(kMinimumSubgroupSize, subgroupInfo.minSubgroupSize));
            std::cout << "subgroup=" << subgroupInfo.minSubgroupSize << "/"
                      << subgroupInfo.maxSubgroupSize << ", select=";
            if (selectedSize < subgroupInfo.minSubgroupSize ||
                selectedSize > subgroupInfo.maxSubgroupSize) {
                std::cout << "invalid(" << selectedSize << ")";
            } else {
                std::cout << selectedSize;
            }
        }
        std::cout << ")\n";
    }

    std::cout << "Select device index: " << std::flush;
    int selectedIndex = -1;
    if (!(std::cin >> selectedIndex) || selectedIndex < 0 ||
        static_cast<std::size_t>(selectedIndex) >= devices.size()) {
        throw std::runtime_error("Invalid device index");
    }
    return static_cast<std::uint32_t>(selectedIndex);
}

class ComputeDemo {
public:
    ~ComputeDemo() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            if (commandPool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, commandPool, nullptr);
            }
            if (descriptorPool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            }
            for (VkPipeline pipelineHandle : pipelines) {
                if (pipelineHandle != VK_NULL_HANDLE) {
                    vkDestroyPipeline(device, pipelineHandle, nullptr);
                }
            }
            if (pipelineLayout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            }
            if (descriptorSetLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            }
            if (shaderModule != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, shaderModule, nullptr);
            }
            if (resultBuffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, resultBuffer, nullptr);
            }
            if (inputBuffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, inputBuffer, nullptr);
            }
            if (resultMemory != VK_NULL_HANDLE) {
                vkFreeMemory(device, resultMemory, nullptr);
            }
            if (inputMemory != VK_NULL_HANDLE) {
                vkFreeMemory(device, inputMemory, nullptr);
            }
            vkDestroyDevice(device, nullptr);
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
    }

    int run(const CommandLineOptions& options) {
        createInstance();
        const auto devices = getPhysicalDevices(instance);
        const std::uint32_t selectedIndex = chooseDeviceIndex(devices, options);
        physicalDevice = devices[selectedIndex];

        const SubgroupSelection subgroupSelection = selectSubgroupSize(
            getSubgroupInfo(physicalDevice), options.simdSize);
        selectedSubgroupSize = subgroupSelection.size;
        useRequiredSubgroupSize = subgroupSelection.useRequiredSubgroupSize;
        if (options.simdSize.has_value() && !useRequiredSubgroupSize) {
            std::cout << "Requested SIMD size " << *options.simdSize
                      << " ignored: required subgroup size is unsupported; using native subgroup "
                      << selectedSubgroupSize << '\n';
        }

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        if (properties.apiVersion < VK_API_VERSION_1_2) {
            throw std::runtime_error("This shader requires a Vulkan 1.2 device");
        }

        const auto input = makeInputData();
        queueFamilyIndex = findComputeQueueFamily();
        createDevice(properties);
        createInputBuffer(input);
        createResultBuffer();
        createDescriptorResources();
        createPipelines();
        return 0;
    }

private:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queueFamilyIndex = 0;
    VkBuffer inputBuffer = VK_NULL_HANDLE;
    VkDeviceMemory inputMemory = VK_NULL_HANDLE;
    VkBuffer resultBuffer = VK_NULL_HANDLE;
    VkDeviceMemory resultMemory = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    std::array<VkPipeline, kPipelineBlockCounts.size()> pipelines{};
    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::uint32_t selectedSubgroupSize = kMinimumSubgroupSize;
    bool useRequiredSubgroupSize = true;
    bool pipelineExecutablePropertiesSupported = false;

    void createInstance() {
        VkApplicationInfo applicationInfo{};
        applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        applicationInfo.pApplicationName = "vulkan_compute_demo";
        applicationInfo.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &applicationInfo;
        checkVk(vkCreateInstance(&createInfo, nullptr, &instance), "vkCreateInstance");
    }

    std::uint32_t findComputeQueueFamily() const {
        std::uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, queueFamilies.data());

        for (std::uint32_t familyIndex = 0; familyIndex < familyCount; ++familyIndex) {
            if (queueFamilies[familyIndex].queueCount > 0 &&
                (queueFamilies[familyIndex].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                return familyIndex;
            }
        }
        throw std::runtime_error("Selected device has no compute queue");
    }

    bool hasDeviceExtension(const char* requiredExtension) const {
        std::uint32_t extensionCount = 0;
        checkVk(vkEnumerateDeviceExtensionProperties(
                    physicalDevice, nullptr, &extensionCount, nullptr),
                "vkEnumerateDeviceExtensionProperties");
        std::vector<VkExtensionProperties> extensions(extensionCount);
        checkVk(vkEnumerateDeviceExtensionProperties(
                    physicalDevice, nullptr, &extensionCount, extensions.data()),
                "vkEnumerateDeviceExtensionProperties");
        for (const auto& extension : extensions) {
            if (std::string(extension.extensionName) == requiredExtension) {
                return true;
            }
        }
        return false;
    }

    void createDevice(const VkPhysicalDeviceProperties& properties) {
        std::vector<const char*> enabledExtensions;
        const bool pipelineExecutableExtensionAvailable =
            hasDeviceExtension(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
        if (pipelineExecutableExtensionAvailable) {
            enabledExtensions.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
        }
        if (properties.apiVersion < VK_API_VERSION_1_3 && useRequiredSubgroupSize) {
            if (!hasDeviceExtension(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME)) {
                throw std::runtime_error(
                    "Selected device lacks VK_KHR_shader_integer_dot_product");
            }
            enabledExtensions.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
            if (!hasDeviceExtension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)) {
                throw std::runtime_error(
                    "Selected device lacks VK_EXT_subgroup_size_control");
            }
            enabledExtensions.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
        }

        VkPhysicalDeviceShaderIntegerDotProductFeatures dotProductFeatures{};
        dotProductFeatures.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;

        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR executablePropertiesFeatures{};
        executablePropertiesFeatures.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;
        if (pipelineExecutableExtensionAvailable) {
            dotProductFeatures.pNext = &executablePropertiesFeatures;
        }

        VkPhysicalDeviceSubgroupSizeControlFeatures subgroupFeatures{};
        subgroupFeatures.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;

        VkPhysicalDeviceVulkan12Features vulkan12Features{};
        vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        if (useRequiredSubgroupSize) {
            subgroupFeatures.pNext = &dotProductFeatures;
            vulkan12Features.pNext = &subgroupFeatures;
        } else {
            vulkan12Features.pNext = &dotProductFeatures;
        }

        VkPhysicalDeviceFeatures2 supportedFeatures{};
        supportedFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        supportedFeatures.pNext = &vulkan12Features;
        vkGetPhysicalDeviceFeatures2(physicalDevice, &supportedFeatures);

        if (!vulkan12Features.shaderInt8 || !dotProductFeatures.shaderIntegerDotProduct ||
            (useRequiredSubgroupSize && !subgroupFeatures.subgroupSizeControl)) {
            throw std::runtime_error(
                "Selected device lacks shaderInt8, integer dot-product, or requested subgroup-size support");
        }
        vulkan12Features.shaderInt8 = VK_TRUE;
        dotProductFeatures.shaderIntegerDotProduct = VK_TRUE;
        if (useRequiredSubgroupSize) {
            subgroupFeatures.subgroupSizeControl = VK_TRUE;
        }

        pipelineExecutablePropertiesSupported =
            pipelineExecutableExtensionAvailable &&
            executablePropertiesFeatures.pipelineExecutableInfo == VK_TRUE;
        if (pipelineExecutablePropertiesSupported) {
            executablePropertiesFeatures.pipelineExecutableInfo = VK_TRUE;
        } else {
            dotProductFeatures.pNext = nullptr;
            const auto extension = std::find(
                enabledExtensions.begin(), enabledExtensions.end(),
                VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
            if (extension != enabledExtensions.end()) {
                enabledExtensions.erase(extension);
            }
        }

        const float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = queueFamilyIndex;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pNext = &vulkan12Features;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pQueueCreateInfos = &queueInfo;
        createInfo.enabledExtensionCount = static_cast<std::uint32_t>(enabledExtensions.size());
        createInfo.ppEnabledExtensionNames =
            enabledExtensions.empty() ? nullptr : enabledExtensions.data();
        checkVk(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device), "vkCreateDevice");
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
    }

    std::uint32_t findHostCoherentMemoryType(std::uint32_t typeFilter) const {
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
        const VkMemoryPropertyFlags requiredFlags =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (std::uint32_t memoryIndex = 0;
             memoryIndex < memoryProperties.memoryTypeCount;
             ++memoryIndex) {
            const bool typeAllowed = (typeFilter & (std::uint32_t{1} << memoryIndex)) != 0;
            const bool flagsAvailable =
                (memoryProperties.memoryTypes[memoryIndex].propertyFlags & requiredFlags) ==
                requiredFlags;
            if (typeAllowed && flagsAvailable) {
                return memoryIndex;
            }
        }
        throw std::runtime_error("No host-visible coherent memory type found");
    }

    void createInputBuffer(const std::array<InputElement, kInvocationCount>& input) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = kInputBufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        checkVk(vkCreateBuffer(device, &bufferInfo, nullptr, &inputBuffer), "vkCreateBuffer");

        VkMemoryRequirements memoryRequirements{};
        vkGetBufferMemoryRequirements(device, inputBuffer, &memoryRequirements);
        VkMemoryAllocateInfo allocationInfo{};
        allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocationInfo.allocationSize = memoryRequirements.size;
        allocationInfo.memoryTypeIndex = findHostCoherentMemoryType(memoryRequirements.memoryTypeBits);
        checkVk(vkAllocateMemory(device, &allocationInfo, nullptr, &inputMemory),
                "vkAllocateMemory");
        checkVk(vkBindBufferMemory(device, inputBuffer, inputMemory, 0), "vkBindBufferMemory");

        void* mappedMemory = nullptr;
        checkVk(vkMapMemory(device, inputMemory, 0, kInputBufferSize, 0, &mappedMemory),
                "vkMapMemory");
        std::memcpy(mappedMemory, input.data(), static_cast<std::size_t>(kInputBufferSize));
        vkUnmapMemory(device, inputMemory);
    }

    void createResultBuffer() {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = kResultBufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        checkVk(vkCreateBuffer(device, &bufferInfo, nullptr, &resultBuffer), "vkCreateBuffer");

        VkMemoryRequirements memoryRequirements{};
        vkGetBufferMemoryRequirements(device, resultBuffer, &memoryRequirements);
        VkMemoryAllocateInfo allocationInfo{};
        allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocationInfo.allocationSize = memoryRequirements.size;
        allocationInfo.memoryTypeIndex = findHostCoherentMemoryType(memoryRequirements.memoryTypeBits);
        checkVk(vkAllocateMemory(device, &allocationInfo, nullptr, &resultMemory), "vkAllocateMemory");
        checkVk(vkBindBufferMemory(device, resultBuffer, resultMemory, 0), "vkBindBufferMemory");
    }

    void createDescriptorResources() {
        std::array<VkDescriptorSetLayoutBinding, 2> storageBindings{};
        storageBindings[0].binding = 0;
        storageBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        storageBindings[0].descriptorCount = 1;
        storageBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        storageBindings[1].binding = 1;
        storageBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        storageBindings[1].descriptorCount = 1;
        storageBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<std::uint32_t>(storageBindings.size());
        layoutInfo.pBindings = storageBindings.data();
        checkVk(vkCreateDescriptorSetLayout(
                    device, &layoutInfo, nullptr, &descriptorSetLayout),
                "vkCreateDescriptorSetLayout");

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 2;
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        checkVk(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool),
                "vkCreateDescriptorPool");

        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = descriptorPool;
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &descriptorSetLayout;
        checkVk(vkAllocateDescriptorSets(device, &allocateInfo, &descriptorSet),
                "vkAllocateDescriptorSets");

        std::array<VkDescriptorBufferInfo, 2> bufferInfos{};
        bufferInfos[0].buffer = inputBuffer;
        bufferInfos[0].range = kInputBufferSize;
        bufferInfos[1].buffer = resultBuffer;
        bufferInfos[1].range = kResultBufferSize;
        std::array<VkWriteDescriptorSet, 2> writeInfos{};
        for (std::size_t binding = 0; binding < writeInfos.size(); ++binding) {
            writeInfos[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeInfos[binding].dstSet = descriptorSet;
            writeInfos[binding].dstBinding = static_cast<std::uint32_t>(binding);
            writeInfos[binding].descriptorCount = 1;
            writeInfos[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writeInfos[binding].pBufferInfo = &bufferInfos[binding];
        }
        vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writeInfos.size()),
                               writeInfos.data(), 0, nullptr);
    }

    std::vector<std::uint32_t> loadShader() const {
        std::ifstream shaderFile(VK_COMPUTE_SHADER_PATH, std::ios::binary | std::ios::ate);
        if (!shaderFile) {
            throw std::runtime_error("Could not open compiled shader: " VK_COMPUTE_SHADER_PATH);
        }
        const std::streamoff fileSize = shaderFile.tellg();
        if (fileSize <= 0 || fileSize % static_cast<std::streamoff>(sizeof(std::uint32_t)) != 0) {
            throw std::runtime_error("Compiled shader has an invalid SPIR-V size");
        }

        std::vector<std::uint32_t> code(
            static_cast<std::size_t>(fileSize) / sizeof(std::uint32_t));
        shaderFile.seekg(0, std::ios::beg);
        shaderFile.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(fileSize));
        if (!shaderFile) {
            throw std::runtime_error("Could not read compiled shader");
        }
        return code;
    }

    void createPipelines() {
        const std::vector<std::uint32_t> code = loadShader();
        VkShaderModuleCreateInfo shaderInfo{};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.codeSize = code.size() * sizeof(std::uint32_t);
        shaderInfo.pCode = code.data();
        checkVk(vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule),
                "vkCreateShaderModule");

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descriptorSetLayout;
        checkVk(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout),
                "vkCreatePipelineLayout");

        std::cout << "Selected subgroup/SIMD size: " << selectedSubgroupSize << '\n';
        for (std::size_t pipelineIndex = 0;
             pipelineIndex < kPipelineBlockCounts.size();
             ++pipelineIndex) {
            pipelines[pipelineIndex] = createPipeline(kPipelineBlockCounts[pipelineIndex]);
            printPipelineStatistics(
            pipelines[pipelineIndex], kPipelineBlockCounts[pipelineIndex]);
        }
        }

        VkPipeline createPipeline(std::uint32_t blockCount) {
        VkPipelineShaderStageCreateInfo stageInfo{};
        stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = shaderModule;
        stageInfo.pName = "main";
        std::array<std::uint32_t, 2> specializationData = {
            selectedSubgroupSize,
            blockCount};
        std::array<VkSpecializationMapEntry, 2> specializationEntries{};
        specializationEntries[0].constantID = 0;
        specializationEntries[0].offset = 0;
        specializationEntries[0].size = sizeof(specializationData[0]);
        specializationEntries[1].constantID = 1;
        specializationEntries[1].offset = sizeof(specializationData[0]);
        specializationEntries[1].size = sizeof(specializationData[1]);
        VkSpecializationInfo specializationInfo{};
        specializationInfo.mapEntryCount = static_cast<std::uint32_t>(specializationEntries.size());
        specializationInfo.pMapEntries = specializationEntries.data();
        specializationInfo.dataSize = specializationData.size() * sizeof(specializationData[0]);
        specializationInfo.pData = specializationData.data();
        stageInfo.pSpecializationInfo = &specializationInfo;
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSizeInfo{};
        subgroupSizeInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
        subgroupSizeInfo.requiredSubgroupSize = selectedSubgroupSize;
        stageInfo.pNext = useRequiredSubgroupSize ? &subgroupSizeInfo : nullptr;
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.flags = pipelineExecutablePropertiesSupported
                                 ? VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR
                                 : 0;
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = pipelineLayout;
        VkPipeline pipelineHandle = VK_NULL_HANDLE;
        checkVk(vkCreateComputePipelines(
                    device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipelineHandle),
                "vkCreateComputePipelines");
        return pipelineHandle;
    }

    void printPipelineStatistics(VkPipeline pipelineHandle, std::uint32_t blockCount) const {
        if (!pipelineExecutablePropertiesSupported) {
            std::cout << "Pipeline executable statistics unavailable for BM=" << blockCount
                      << " (VK_KHR_pipeline_executable_properties is unavailable)\n";
            return;
        }

        const auto getProperties = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
            vkGetDeviceProcAddr(device, "vkGetPipelineExecutablePropertiesKHR"));
        const auto getStatistics = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
            vkGetDeviceProcAddr(device, "vkGetPipelineExecutableStatisticsKHR"));
        if (getProperties == nullptr || getStatistics == nullptr) {
            std::cout << "Pipeline executable statistics unavailable for BM=" << blockCount
                      << " (query functions are unavailable)\n";
            return;
        }

        VkPipelineInfoKHR pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR;
        pipelineInfo.pipeline = pipelineHandle;
        std::uint32_t executableCount = 0;
        VkResult queryResult = getProperties(device, &pipelineInfo, &executableCount, nullptr);
        if (queryResult != VK_SUCCESS) {
            std::cout << "Pipeline executable statistics unavailable for BM=" << blockCount
                  << " (properties query returned " << static_cast<int>(queryResult) << ")\n";
            return;
        }
        std::vector<VkPipelineExecutablePropertiesKHR> executables(executableCount);
        for (auto& executable : executables) {
            executable.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
        }
        queryResult = getProperties(device, &pipelineInfo, &executableCount, executables.data());
        if (queryResult != VK_SUCCESS) {
            std::cout << "Pipeline executable statistics unavailable for BM=" << blockCount
                      << " (properties query returned " << static_cast<int>(queryResult) << ")\n";
            return;
        }

        std::cout << "Pipeline executable report (BM=" << blockCount << "):\n";
        for (std::uint32_t executableIndex = 0;
             executableIndex < executableCount;
             ++executableIndex) {
            VkPipelineExecutableInfoKHR executableInfo{};
            executableInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR;
            executableInfo.pipeline = pipelineHandle;
            executableInfo.executableIndex = executableIndex;

            std::uint32_t statisticCount = 0;
            queryResult = getStatistics(device, &executableInfo, &statisticCount, nullptr);
            if (queryResult != VK_SUCCESS) {
                std::cout << "Pipeline executable statistics unavailable for BM=" << blockCount
                          << " (statistics query returned " << static_cast<int>(queryResult) << ")\n";
                return;
            }
            std::vector<VkPipelineExecutableStatisticKHR> statistics(statisticCount);
            for (auto& statistic : statistics) {
                statistic.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
            }
            queryResult = getStatistics(device, &executableInfo, &statisticCount, statistics.data());
            if (queryResult != VK_SUCCESS) {
                std::cout << "Pipeline executable statistics unavailable for BM=" << blockCount
                          << " (statistics query returned " << static_cast<int>(queryResult) << ")\n";
                return;
            }

            std::cout << "  Executable " << executableIndex << ": "
                      << executables[executableIndex].name << '\n';
            for (const auto& statistic : statistics) {
                const std::string statisticName = statistic.name;
                std::cout << "    " << statisticName << ": ";
                printStatisticValue(statistic);
                std::cout << '\n';
            }
        }
    }

    bool dispatchAndReadResults(const ExpectedResults& expected) {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndex;
        checkVk(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool),
                "vkCreateCommandPool");

        VkCommandBufferAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocateInfo.commandPool = commandPool;
        allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = 1;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        checkVk(vkAllocateCommandBuffers(device, &allocateInfo, &commandBuffer),
                "vkAllocateCommandBuffers");

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        checkVk(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines.back());
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        vkCmdDispatch(commandBuffer, 1, 1, 1);

        VkBufferMemoryBarrier hostReadBarrier{};
        hostReadBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        hostReadBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        hostReadBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        hostReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostReadBarrier.buffer = resultBuffer;
        hostReadBarrier.offset = 0;
        hostReadBarrier.size = kResultBufferSize;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT,
                             0, 0, nullptr, 1, &hostReadBarrier, 0, nullptr);
        checkVk(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        checkVk(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE), "vkQueueSubmit");
        checkVk(vkQueueWaitIdle(queue), "vkQueueWaitIdle");

        void* mappedMemory = nullptr;
        checkVk(vkMapMemory(device, resultMemory, 0, kResultBufferSize, 0, &mappedMemory),
                "vkMapMemory");
        const auto* values = static_cast<const std::uint8_t*>(mappedMemory);
        vkUnmapMemory(device, resultMemory);

        bool resultsMatch = true;
        for (std::size_t index = 0; index < kInvocationCount; ++index) {
            resultsMatch = resultsMatch && values[index] == expected.values[index];
        }
        std::cout << "VK 16 invocations: " << (resultsMatch ? "MATCH" : "MISMATCH") << '\n';
        return resultsMatch;
    }
};

}

int main(int argc, char* argv[]) {
    try {
        const auto options = parseCommandLine(argc, argv);
        ComputeDemo demo;
        return demo.run(options);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}