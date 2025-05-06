cmake_minimum_required(VERSION 3.7)

project(PasswordCracker)

add_definitions(-DCL_TARGET_OPENCL_VERSION=300)

find_package(OpenCL REQUIRED)

file(GLOB SRC_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Werror) 

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/kernels ${CMAKE_BINARY_DIR}/kernels
)
