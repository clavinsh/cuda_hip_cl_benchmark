# Daļa no faila ņemta šeit: https://github.com/ROCm/rocm-examples/blob/develop/HIP-Basic/saxpy/CMakeLists.txt

cmake_minimum_required(VERSION 3.7)
project(HipPwCracker LANGUAGES CXX)

set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")
set(GPU_RUNTIMES "HIP" "CUDA")
set_property(CACHE GPU_RUNTIME PROPERTY STRINGS ${GPU_RUNTIMES})
if(NOT "${GPU_RUNTIME}" IN_LIST GPU_RUNTIMES)
    message(FATAL_ERROR "GPU_RUNTIME must be either HIP or CUDA.")
endif()

enable_language(${GPU_RUNTIME})

set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)

if(WIN32)
    set(ROCM_ROOT "$ENV{HIP_PATH}" CACHE PATH "Root directory of the ROCm installation")
else()
    set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
endif()
list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

find_package(spdlog REQUIRED)

file(GLOB SRC_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB GPU_SRC_FILES "${CMAKE_SOURCE_DIR}/src/*.hip")

add_executable(${PROJECT_NAME} ${SRC_FILES} ${GPU_SRC_FILES})

foreach(gpu_file ${GPU_SRC_FILES})
    set_source_files_properties(${gpu_file} PROPERTIES LANGUAGE ${GPU_RUNTIME})
endforeach()

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
)

if(GPU_RUNTIME STREQUAL "CUDA")
    target_include_directories(${PROJECT_NAME} PRIVATE ${ROCM_ROOT}/include)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE 
    spdlog::spdlog
)

target_compile_options(${PROJECT_NAME} PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>
)

if(GPU_RUNTIME STREQUAL "HIP")
    set(GPU_TARGETS "gfx900;gfx906;gfx908;gfx90a;gfx1030" CACHE STRING "AMD GPU architectures")
    set_property(TARGET ${PROJECT_NAME} PROPERTY HIP_ARCHITECTURES ${GPU_TARGETS})
elseif(GPU_RUNTIME STREQUAL "CUDA")
    set(GPU_TARGETS "60;70;75;80;86" CACHE STRING "NVIDIA GPU architectures")
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES ${GPU_TARGETS})
endif()


