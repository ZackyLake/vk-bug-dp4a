cmake_minimum_required(VERSION 3.20)
project(vulkan_compute_demo LANGUAGES CXX)

find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE
    NAMES glslc
    HINTS "$ENV{VULKAN_SDK}/Bin"
    REQUIRED)

set(SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shader.comp")
set(SHADER_BINARY "${CMAKE_CURRENT_BINARY_DIR}/shader.comp.spv")

add_custom_command(
    OUTPUT "${SHADER_BINARY}"
    COMMAND "${GLSLC_EXECUTABLE}" --target-env=vulkan1.2 "${SHADER_SOURCE}" -o "${SHADER_BINARY}"
    DEPENDS "${SHADER_SOURCE}"
    VERBATIM)

add_custom_target(compile_shader DEPENDS "${SHADER_BINARY}")

add_executable(vulkan_compute_demo main.cpp)
add_dependencies(vulkan_compute_demo compile_shader)
target_compile_features(vulkan_compute_demo PRIVATE cxx_std_17)
target_compile_definitions(vulkan_compute_demo PRIVATE
    VK_COMPUTE_SHADER_PATH=\"${SHADER_BINARY}\")
target_link_libraries(vulkan_compute_demo PRIVATE Vulkan::Vulkan)