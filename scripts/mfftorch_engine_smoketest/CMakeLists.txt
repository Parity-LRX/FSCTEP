cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mfftorch_engine_smoketest CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_executable(mfftorch_engine_smoketest main.cpp
  ../lammps_user_mfftorch/src/USER-MFFTORCH/mff_torch_engine.cpp
)
target_include_directories(mfftorch_engine_smoketest PRIVATE
  ../lammps_user_mfftorch/src/USER-MFFTORCH
)
target_link_libraries(mfftorch_engine_smoketest "${TORCH_LIBRARIES}")

set_property(TARGET mfftorch_engine_smoketest PROPERTY CXX_STANDARD 17)

