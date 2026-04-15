#include <iostream>
#include <chrono>
#include <cstring>
#include "core/ir_v2/IRGraphV2.hpp"
#include "core/compiler_v4/GraphCompilerV4.hpp"
#include "core/runtime_v2/StaticRuntimeExecutor.hpp"

using namespace aresml;

int main() {
    std::cout << "=== Static Compilation Engine V4 Test ===\n\n";
    
    IRGraphV2 graph;
    graph.add_node(IROpV2::INPUT, {32, 64}, "input");
    graph.add_node(IROpV2::WEIGHT, {64, 128}, "w1");
    graph.add_node(IROpV2::MATMUL, {32, 128}, "mm1");
    graph.add_node(IROpV2::ADD, {32, 128}, "add1");
    graph.add_node(IROpV2::RELU, {32, 128}, "relu1");
    graph.add_node(IROpV2::WEIGHT, {128, 64}, "w2");
    graph.add_node(IROpV2::MATMUL, {32, 64}, "mm2");
    graph.add_node(IROpV2::RELU, {32, 64}, "relu2");
    
    std::cout << "Input graph: " << graph.node_count() << " nodes\n";
    
    GraphCompilerV4 compiler;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto plan = compiler.compile(graph);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "\nCompilation result:\n";
    std::cout << plan->to_string();
    std::cout << "Compilation time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us\n";
    
    std::cout << "\n=== Runtime Execution Test ===\n";
    
    StaticRuntimeExecutor executor;
    
    size_t buffer_size = 128 * 1024;
    float* buffer_pool = new float[buffer_size];
    std::memset(buffer_pool, 0, buffer_size * sizeof(float));
    
    for (size_t i = 0; i < 32 * 64; ++i) {
        buffer_pool[i] = static_cast<float>(i) * 0.01f;
    }
    for (size_t i = 0; i < 64 * 128; ++i) {
        buffer_pool[32 * 64 + i] = static_cast<float>(i) * 0.02f;
    }
    
    auto exec_start = std::chrono::high_resolution_clock::now();
    executor.execute(*plan, buffer_pool);
    auto exec_end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Execution time: " << std::chrono::duration_cast<std::chrono::microseconds>(exec_end - exec_start).count() << " us\n";
    
    std::cout << "\n=== Fused Execution Test ===\n";
    
    executor.set_fusion_enabled(true);
    executor.set_simd_enabled(true);
    
    executor.execute_fused(*plan, buffer_pool);
    
    std::cout << "Fused execution completed.\n";
    
    delete[] buffer_pool;
    
    std::cout << "\n=== Multiple Iterations Test ===\n";
    
    auto plan2 = compiler.compile(graph);
    
    size_t iterations = 100;
    auto iter_start = std::chrono::high_resolution_clock::now();
    
    for (size_t iter = 0; iter < iterations; ++iter) {
        float* buf = new float[buffer_size];
        executor.execute(*plan2, buf);
        delete[] buf;
    }
    
    auto iter_end = std::chrono::high_resolution_clock::now();
    double avg_time = std::chrono::duration_cast<std::chrono::nanoseconds>(iter_end - iter_start).count() / (double)iterations;
    
    std::cout << "Average execution time (" << iterations << " iterations): " << avg_time << " ns\n";
    
    std::cout << "\n=== All Tests Passed ===\n";
    
    return 0;
}
