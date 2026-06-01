cmake_minimum_required(VERSION 3.18)
project(mff_cg_repro LANGUAGES CXX)

find_package(Torch REQUIRED)

set(ENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src/USER-MFFTORCH)

add_executable(mff_cg_repro
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${ENGINE_DIR}/mff_torch_engine.cpp
)

target_include_directories(mff_cg_repro PRIVATE ${ENGINE_DIR} ${TORCH_INCLUDE_DIRS})
target_link_libraries(mff_cg_repro PRIVATE "${TORCH_LIBRARIES}")

if(TORCH_CXX_FLAGS)
  separate_arguments(_TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(mff_cg_repro PRIVATE ${_TORCH_CXX_FLAGS_LIST})
endif()

set_property(TARGET mff_cg_repro PROPERTY CXX_STANDARD 17)
