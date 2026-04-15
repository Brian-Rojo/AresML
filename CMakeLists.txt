cmake_minimum_required(VERSION 3.16)
project(AresML VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_OPENCL "Enable OpenCL backend" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

set(CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core)
set(CORE_IR_V2_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core/ir_v2)
set(CORE_IR_V3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core/ir_v3)
set(CORE_FUSION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core/fusion)
set(CORE_COMPILER_V3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core/compiler_v3)
set(CORE_KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core/kernel_specialization)
set(CORE_LAYOUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core/layout_optimizer)
set(CORE_EXECUTION_V3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core/execution_v3)
set(CORE_RUNTIME_V3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core/runtime_v3)
set(CORE_MEMORY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core/memory)
set(CORE_SCHEDULER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core/scheduler)
set(CORE_PREFETCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core/prefetch)
set(CORE_SIMD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core/simd)
set(CORE_THREADING_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core/threading)
set(CORE_MEMORY_GRAPH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core/memory_graph)
set(CORE_COMPILER_V4_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core/compiler_v4)
set(CORE_RUNTIME_V2_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core/runtime_v2)
set(CORE_MEMORY_PLAN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core/memory_plan)
set(CORE_LOWERING_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core/lowering)
set(NN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nn)
set(NN_TRANSFORMER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nn/transformer)
set(ENGINE_GPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/engine/gpt)
set(OPTIM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/optim)
set(LOSS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/loss)
set(BACKEND_CPU_DIR ${CMAKE_CURRENT_SOURCE_DIR}/backend_cpu)
set(OPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ops)
set(UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utils)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CORE_DIR})
include_directories(${CORE_IR_V2_DIR})
include_directories(${CORE_IR_V3_DIR})
include_directories(${CORE_FUSION_DIR})
include_directories(${CORE_COMPILER_V3_DIR})
include_directories(${CORE_KERNEL_DIR})
include_directories(${CORE_LAYOUT_DIR})
include_directories(${CORE_EXECUTION_V3_DIR})
include_directories(${CORE_RUNTIME_V3_DIR})
include_directories(${CORE_MEMORY_DIR})
include_directories(${CORE_SCHEDULER_DIR})
include_directories(${CORE_PREFETCH_DIR})
include_directories(${CORE_SIMD_DIR})
include_directories(${CORE_THREADING_DIR})
include_directories(${CORE_MEMORY_GRAPH_DIR})
include_directories(${CORE_COMPILER_V4_DIR})
include_directories(${CORE_RUNTIME_V2_DIR})
include_directories(${CORE_MEMORY_PLAN_DIR})
include_directories(${CORE_LOWERING_DIR})
include_directories(${NN_DIR})
include_directories(${NN_TRANSFORMER_DIR})
include_directories(${ENGINE_GPT_DIR})
include_directories(${OPTIM_DIR})
include_directories(${LOSS_DIR})
include_directories(${BACKEND_CPU_DIR})
include_directories(${OPS_DIR})
include_directories(${UTILS_DIR})

set(HEADER_SOURCES
    core/Tensor.hpp
    core/Autograd.hpp
    core/ir_v2/IRNodeV2.hpp
    core/ir_v2/IRGraphV2.hpp
    core/ir_v3/IRNodeV3.hpp
    core/ir_v3/IRGraphV3.hpp
    core/fusion/PatternMatcher.hpp
    core/fusion/FusionGroups.hpp
    core/fusion/FusionEngineV2.hpp
    core/compiler_v3/GraphCompilerV3.hpp
    core/compiler_v3/GraphLoweringV3.hpp
    core/compiler_v3/PassManagerV3.hpp
    core/kernel_specialization/KernelDB.hpp
    core/layout_optimizer/LayoutAnalyzer.hpp
    core/execution_v3/ExecutionPlan.hpp
    core/runtime_v3/PlanExecutor.hpp
    core/memory/CachePlanner.hpp
    core/memory/MemoryPlan.hpp
    core/runtime/FusedKernelExecutor.hpp
    core/scheduler/SchedulerV2.hpp
    core/prefetch/PrefetchEngine.hpp
    core/simd/SimdKernelsV2.hpp
    core/memory_graph/MemoryGraphEngine.hpp
    core/compiler_v4/GraphCompilerV4.hpp
    core/runtime_v2/StaticRuntimeExecutor.hpp
    nn/Linear.hpp
    nn/RMSNorm.hpp
    nn/Attention.hpp
    nn/FFN.hpp
    nn/TransformerBlock.hpp
    nn/LMHead.hpp
    optim/SGD.hpp
    optim/Adam.hpp
    loss/CrossEntropy.hpp
    loss/MSELoss.hpp
    backend_cpu/Matmul.hpp
    backend_cpu/Softmax.hpp
    ops/Ops.hpp
    utils/GradientCheck.hpp
)

add_library(aresml INTERFACE)
target_sources(aresml INTERFACE ${HEADER_SOURCES})

add_executable(train examples/train.cpp)
add_executable(train_simple examples/train_simple.cpp)
add_executable(test_linear examples/test_linear.cpp)
add_executable(test_attention examples/test_attention.cpp)
add_executable(test_attention_full examples/test_attention_full.cpp)
target_link_libraries(test_attention_full aresml)
add_executable(test_block examples/test_block.cpp)
add_executable(test_forward examples/test_forward.cpp)
add_executable(train_simple_linear examples/train_simple_linear.cpp)
add_executable(debug_train examples/debug_train.cpp)
add_executable(test_stable examples/test_stable.cpp)
add_executable(test_debug examples/test_debug.cpp)
add_executable(test_quick examples/test_quick.cpp)
add_executable(test_v03 examples/test_v03.cpp)
add_executable(test_ce examples/test_ce.cpp)
add_executable(test_mse examples/test_mse.cpp)
add_executable(test_cef examples/test_cef.cpp)
add_executable(test_ceb examples/test_ceb.cpp)
add_executable(test_simple examples/test_simple.cpp)
add_executable(test_50 examples/test_50.cpp)
add_executable(test_100 examples/test_100.cpp)
add_executable(test_gradient_check examples/test_gradient_check.cpp)
add_executable(test_gradient_debug examples/test_gradient_debug.cpp)
add_executable(test_logsoftmax_debug examples/test_logsoftmax_debug.cpp)
add_executable(test_ce_gradient_debug examples/test_ce_gradient_debug.cpp)
add_executable(test_profile examples/test_profile.cpp)
add_executable(test_audit examples/test_audit.cpp)
target_link_libraries(test_audit aresml)
target_link_libraries(test_gradient_check aresml)
target_link_libraries(test_gradient_debug aresml)
target_link_libraries(test_logsoftmax_debug aresml)
target_link_libraries(test_ce_gradient_debug aresml)
target_link_libraries(test_profile aresml)
target_link_libraries(test_100 aresml)
target_link_libraries(test_ceb aresml)
target_link_libraries(test_stable aresml)
target_link_libraries(test_debug aresml)
target_link_libraries(test_quick aresml)
target_link_libraries(test_block aresml)
target_link_libraries(debug_train aresml)
target_link_libraries(test_forward aresml)
target_link_libraries(train aresml)
target_link_libraries(train_simple aresml)
target_link_libraries(train_simple_linear aresml)
target_link_libraries(test_linear aresml)
target_link_libraries(test_attention aresml)

add_executable(test_gpt_forward examples/test_gpt_forward.cpp)
add_executable(test_gpt_overfit examples/test_gpt_overfit.cpp)
add_executable(test_gpt_generation examples/test_gpt_generation.cpp)
target_link_libraries(test_gpt_forward aresml)
target_link_libraries(test_gpt_overfit aresml)
target_link_libraries(test_gpt_generation aresml)

add_executable(test_simple_linear examples/test_simple_linear.cpp)
target_link_libraries(test_simple_linear aresml)

add_executable(test_autograd_gradients test_autograd_gradients.cpp)
target_link_libraries(test_autograd_gradients aresml)

add_executable(test_autograd_core examples/test_autograd_core.cpp)
target_link_libraries(test_autograd_core aresml)

add_executable(test_simd_benchmark examples/test_simd_benchmark.cpp)
target_link_libraries(test_simd_benchmark aresml)

add_executable(test_fusion examples/test_fusion.cpp)
target_link_libraries(test_fusion aresml)

add_executable(test_fusion_v2 examples/test_fusion_v2.cpp)
target_link_libraries(test_fusion_v2 aresml)

add_executable(test_compile_graph examples/test_compile_graph.cpp)
target_link_libraries(test_compile_graph aresml)

add_executable(test_memory_graph examples/test_memory_graph.cpp)
target_link_libraries(test_memory_graph aresml)

add_executable(test_phase17_static examples/test_phase17_static.cpp)
target_link_libraries(test_phase17_static aresml)

enable_testing()
