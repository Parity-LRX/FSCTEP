cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(libtorch_smoketest CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_executable(libtorch_smoketest main.cpp)
target_link_libraries(libtorch_smoketest "${TORCH_LIBRARIES}")

# 避免某些平台上的符号丢失/链接问题
set_property(TARGET libtorch_smoketest PROPERTY CXX_STANDARD 17)

