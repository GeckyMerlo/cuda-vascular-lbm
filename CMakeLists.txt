cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
endif()

project(vascular_lbm LANGUAGES CXX CUDA)

set(CUDA_BLOCK_SIZE 256 CACHE STRING "CUDA thread block size for 1D kernel launches")

add_executable(vascular_lbm
    src/main.cu
    src/lbm/lbm_constants.cu
    src/lbm/lbm_system.cpp
    src/space/space_system.cpp
    src/particles/particle_system.cpp
    src/forces/force_model.cpp
    src/msh_utils/mesh_reader.cpp
)

set_source_files_properties(
    src/lbm/lbm_system.cpp
    src/space/space_system.cpp
    src/particles/particle_system.cpp
    src/forces/force_model.cpp
    PROPERTIES LANGUAGE CUDA
)

target_include_directories(vascular_lbm PRIVATE src)
target_compile_definitions(vascular_lbm PRIVATE CUDA_BLOCK_SIZE=${CUDA_BLOCK_SIZE})

set_target_properties(vascular_lbm PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)
