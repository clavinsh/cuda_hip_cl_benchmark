cmake_minimum_required(VERSION 3.7)
project(CudaPwCracker LANGUAGES CXX CUDA)

# Find spdlog package
find_package(spdlog REQUIRED)

# Collect source files
file(GLOB SRC_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB CUDA_SRC_FILES "${CMAKE_SOURCE_DIR}/src/*.cu")

# Set CUDA architecture(s) based on your target GPU
set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86)  # For Pascal, Volta, Turing, Ampere

# Create executable with both CPU and GPU sources
add_executable(${PROJECT_NAME} ${SRC_FILES} ${CUDA_SRC_FILES})

# Set include directories
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE 
    spdlog::spdlog
)

# Set compiler options
target_compile_options(${PROJECT_NAME} PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Set C++ standard (adjust as needed)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
