cmake_minimum_required(VERSION 3.7)

project(PasswordCracker)

find_package(OpenCL REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Werror -g3) #-g3 priekš debug
