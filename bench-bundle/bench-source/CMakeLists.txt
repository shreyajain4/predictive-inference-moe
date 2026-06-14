set(TARGET llama-moe-predictor-bench)

# Per-platform layer-stream/cache sources.
set(MOE_CACHE_SOURCES)
if (APPLE)
    list(APPEND MOE_CACHE_SOURCES
        moe_metal_expert_cache.mm
        moe_layer_stream.mm
    )
elseif (GGML_CUDA)
    list(APPEND MOE_CACHE_SOURCES
        moe_cuda_layer_stream.cu
        moe_cuda_expert_cache.cu
    )
endif()

add_executable(${TARGET} moe-predictor-bench.cpp moe_predictor.cpp ${MOE_CACHE_SOURCES})
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE .)

if (APPLE)
    set_source_files_properties(moe_metal_expert_cache.mm moe_layer_stream.mm PROPERTIES
        COMPILE_FLAGS "-fobjc-arc")
    target_link_libraries(${TARGET} PRIVATE
        "-framework Foundation"
        "-framework Metal")
    target_compile_definitions(${TARGET} PRIVATE
        MOE_METAL_CACHE_AVAILABLE=1
        MOE_LAYER_STREAM_AVAILABLE=1)
elseif (GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    set_source_files_properties(moe_cuda_layer_stream.cu moe_cuda_expert_cache.cu PROPERTIES
        LANGUAGE CUDA)
    target_link_libraries(${TARGET} PRIVATE CUDA::cudart /usr/lib/x86_64-linux-gnu/libstdc++.so.6 pthread)
    set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)
    target_compile_definitions(${TARGET} PRIVATE
        MOE_LAYER_STREAM_AVAILABLE=1
        MOE_LAYER_STREAM_BACKEND_CUDA=1
        MOE_CUDA_EXPERT_CACHE_AVAILABLE=1)
endif()
