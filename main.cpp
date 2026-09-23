#include <vulkan/vulkan.h>

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
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
constexpr std::uint32_t kPackedWeights = 0x00000801u;
constexpr VkDeviceSize kResultBufferSize = 2 * sizeof(std::uint32_t);

struct ExpectedResults {
    std::uint32_t func1;
    std::uint32_t func2;
};

std::array<std::uint32_t, 4> unpackBytes(std::uint32_t packed) {
    return {
        packed & 0xffu,
        (packed >> 8) & 0xffu,
        (packed >> 16) & 0xffu,
        (packed >> 24) & 0xffu};
}

ExpectedResults printInputAndExpectedResults() {
    const auto inputBytes = unpackBytes(kPackedInput);
    const auto weightBytes = unpackBytes(kPackedWeights);
    const std::uint32_t expectedFunc1 = inputBytes[0] + 8u * inputBytes[1];
    std::uint32_t dotProduct = 0;
    for (std::size_t componentIndex = 0; componentIndex < inputBytes.size(); ++componentIndex) {
        dotProduct += inputBytes[componentIndex] * weightBytes[componentIndex];
    }
    const std::uint32_t expectedFunc2 = dotProduct;

    std::cout << "packed = 0x" << std::hex << std::setw(8) << std::setfill('0')
              << kPackedInput << std::dec << std::setfill(' ') << " = u8["
              << inputBytes[0] << ", " << inputBytes[1] << ", "
              << inputBytes[2] << ", " << inputBytes[3] << "]\n"
              << "dp4a with 0x" << std::hex << std::setw(8) << std::setfill('0')
              << kPackedWeights << std::dec << std::setfill(' ') << " = u8["
              << weightBytes[0] << ", " << weightBytes[1] << ", "
              << weightBytes[2] << ", " << weightBytes[3] << "]\n"
              << "expected = " << expectedFunc1 << '\n';

    return {expectedFunc1, expectedFunc2};
}

void checkVk(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with VkResult " +
                                 std::to_string(static_cast<int>(result)));
    }
}

std::optional<std::uint32_t> parseDeviceIndex(int argc, char* argv[]) {
    std::optional<std::uint32_t> requestedIndex;
    for (int argumentIndex = 1; argumentIndex < argc; ++argumentIndex) {
        if (std::string(argv[argumentIndex]) != "-i" || requestedIndex.has_value() ||
            argumentIndex + 1 >= argc) {
            throw std::runtime_error("Usage: vulkan_compute_demo [-i device_index]");
        }

        const std::string value = argv[++argumentIndex];
        std::uint32_t parsedIndex = 0;
        const auto parseResult = std::from_chars(
            value.data(), value.data() + value.size(), parsedIndex);
        if (parseResult.ec != std::errc{} || parseResult.ptr != value.data() + value.size()) {
            throw std::runtime_error("Device index must be a non-negative integer");
        }
        requestedIndex = parsedIndex;
    }
    return requestedIndex;
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

std::uint32_t chooseDeviceIndex(
    const std::vector<VkPhysicalDevice>& devices,
    const std::optional<std::uint32_t>& requestedIndex) {
    if (requestedIndex.has_value()) {
        if (*requestedIndex >= devices.size()) {
            throw std::runtime_error("Device index is out of range");
        }
        return *requestedIndex;
    }

    for (std::size_t deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(devices[deviceIndex], &properties);
        std::cout << deviceIndex << ": " << properties.deviceName << '\n';
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
            if (pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, pipeline, nullptr);
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
            if (resultMemory != VK_NULL_HANDLE) {
                vkFreeMemory(device, resultMemory, nullptr);
            }
            vkDestroyDevice(device, nullptr);
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
    }

    int run(const std::optional<std::uint32_t>& requestedIndex) {
        createInstance();
        const auto devices = getPhysicalDevices(instance);
        const std::uint32_t selectedIndex = chooseDeviceIndex(devices, requestedIndex);
        physicalDevice = devices[selectedIndex];

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        if (properties.apiVersion < VK_API_VERSION_1_2) {
            throw std::runtime_error("This shader requires a Vulkan 1.2 device");
        }

        const ExpectedResults expected = printInputAndExpectedResults();
        queueFamilyIndex = findComputeQueueFamily();
        createDevice(properties);
        createResultBuffer();
        createDescriptorResources();
        createPipeline();
        return dispatchAndReadResults(expected) ? 0 : 1;
    }

private:
    struct PushConstants {
        std::uint32_t val;
    };

    static_assert(sizeof(PushConstants) == sizeof(std::uint32_t));

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queueFamilyIndex = 0;
    VkBuffer resultBuffer = VK_NULL_HANDLE;
    VkDeviceMemory resultMemory = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;

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
        if (properties.apiVersion < VK_API_VERSION_1_3) {
            if (!hasDeviceExtension(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME)) {
                throw std::runtime_error(
                    "Selected device lacks VK_KHR_shader_integer_dot_product");
            }
            enabledExtensions.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
        }

        VkPhysicalDeviceShaderIntegerDotProductFeatures dotProductFeatures{};
        dotProductFeatures.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;

        VkPhysicalDeviceVulkan12Features vulkan12Features{};
        vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vulkan12Features.pNext = &dotProductFeatures;

        VkPhysicalDeviceFeatures2 supportedFeatures{};
        supportedFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        supportedFeatures.pNext = &vulkan12Features;
        vkGetPhysicalDeviceFeatures2(physicalDevice, &supportedFeatures);

        if (!vulkan12Features.shaderInt8 || !dotProductFeatures.shaderIntegerDotProduct) {
            throw std::runtime_error(
            "Selected device lacks shaderInt8 or integer dot-product support");
        }
        vulkan12Features.shaderInt8 = VK_TRUE;
        dotProductFeatures.shaderIntegerDotProduct = VK_TRUE;

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
        VkDescriptorSetLayoutBinding storageBinding{};
        storageBinding.binding = 0;
        storageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        storageBinding.descriptorCount = 1;
        storageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &storageBinding;
        checkVk(vkCreateDescriptorSetLayout(
                    device, &layoutInfo, nullptr, &descriptorSetLayout),
                "vkCreateDescriptorSetLayout");

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 1;
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

        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = resultBuffer;
        bufferInfo.offset = 0;
        bufferInfo.range = kResultBufferSize;
        VkWriteDescriptorSet writeInfo{};
        writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeInfo.dstSet = descriptorSet;
        writeInfo.dstBinding = 0;
        writeInfo.descriptorCount = 1;
        writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeInfo.pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(device, 1, &writeInfo, 0, nullptr);
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

    void createPipeline() {
        const std::vector<std::uint32_t> code = loadShader();
        VkShaderModuleCreateInfo shaderInfo{};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.codeSize = code.size() * sizeof(std::uint32_t);
        shaderInfo.pCode = code.data();
        checkVk(vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule),
                "vkCreateShaderModule");

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PushConstants);
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descriptorSetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;
        checkVk(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout),
                "vkCreatePipelineLayout");

        VkPipelineShaderStageCreateInfo stageInfo{};
        stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = shaderModule;
        stageInfo.pName = "main";
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = pipelineLayout;
        checkVk(vkCreateComputePipelines(
                    device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline),
                "vkCreateComputePipelines");
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
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        const PushConstants pushConstants{kPackedInput};
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pushConstants), &pushConstants);
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
        const auto* values = static_cast<const std::uint32_t*>(mappedMemory);
        const std::uint32_t firstResult = values[0];
        const std::uint32_t secondResult = values[1];
        vkUnmapMemory(device, resultMemory);

        const bool func1Matches = firstResult == expected.func1;
        const bool func2Matches = secondResult == expected.func2;
        const bool resultsMatch =
            func1Matches && func2Matches && firstResult == secondResult;
        std::cout << "VK func1 (MULADD) = " << firstResult << " ("
                  << (func1Matches ? "PASS" : "FAIL") << ")\n"
                  << "VK func2 (DP4A) = " << secondResult << " ("
                  << (func2Matches ? "PASS" : "FAIL") << ")\n"
                  << (resultsMatch ? "MATCH" : "MISMATCH") << '\n';
        return resultsMatch;
    }
};

}

int main(int argc, char* argv[]) {
    try {
        const auto requestedIndex = parseDeviceIndex(argc, argv);
        ComputeDemo demo;
        return demo.run(requestedIndex);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}