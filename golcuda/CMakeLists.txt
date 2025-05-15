cmake_minimum_required(VERSION 3.7)
project(GameOfLifeCuda LANGUAGES CXX CUDA)

find_package(spdlog REQUIRED)

file(GLOB SRC_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB CUDA_SRC_FILES "${CMAKE_SOURCE_DIR}/src/*.cu")

set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86)  # Pascal, Volta, Turing, Ampere

add_executable(${PROJECT_NAME} ${SRC_FILES} ${CUDA_SRC_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    spdlog::spdlog
)

target_compile_options(${PROJECT_NAME} PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror -O3>
    $<$<COMPILE_LANGUAGE:CUDA>: -O3>
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
