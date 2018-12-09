cmake_minimum_required(VERSION 2.6)
project(gpu_seg LANGUAGES CUDA CXX)

set (CMAKE_CXX_STANDARD 11)

# find CUDA
find_package( CUDA 10.0 EXACT REQUIRED)

if (CUDA_FOUND)
    message("CUDA found")
    SET(NVGRAPH_INCLUDE_DIRS "/usr/local/cuda-10.0/targets/x86_64-linux/include" ${NVGRAPH_INCLUDE_DIRS})
    #include(FindCUDA)
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
    INCLUDE_DIRECTORIES(${NVGRAPH_INCLUDE_DIRS})
else()
    message("CUDA not found")
endif()

# add_executable(segmentacao_sequencial main.cpp imagem.cpp)
add_executable(gpu_seg main.cu imagem.cpp)
target_link_libraries(gpu_seg ${CUDA_LIBRARIES} nvgraph)

install(TARGETS gpu_seg RUNTIME DESTINATION bin)
