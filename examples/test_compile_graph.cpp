#include <iostream>
#include <chrono>
#include "core/ir_v2/IRGraphV2.hpp"
#include "core/compiler_v3/GraphCompilerV3.hpp"
#include "core/runtime_v3/PlanExecutor.hpp"

using namespace aresml;

int main() {
    std::cout << "=== GraphCompilerV3 Test ===\n\n";
    
    IRGraphV2 graph;
    graph.add_node(IROpV2::INPUT, {32, 64}, "input");
    graph.add_node(IROpV2::WEIGHT, {64, 128}, "w1");
    graph.add_node(IROpV2::MATMUL, {32, 128}, "mm1");
    graph.add_node(IROpV2::BIAS_ADD, {32, 128}, "bias1");
    graph.add_node(IROpV2::RELU, {32, 128}, "relu1");
    
    std::cout << "Original graph: " << graph.node_count() << " nodes\n";
    
    GraphCompilerV3 compiler;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = compiler.compile(graph);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "\nCompilation result:\n";
    std::cout << "  Graph hash: " << result.graph_hash << "\n";
    std::cout << "  From cache: " << (result.from_cache ? "yes" : "no") << "\n";
    std::cout << "  Estimated speedup: " << result.estimated_speedup << "x\n";
    std::cout << "  Compilation time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us\n";
    
    if (result.plan) {
        std::cout << "\n" << result.plan->to_string();
    }
    
    std::cout << "\n=== Cache Test ===\n";
    
    auto start2 = std::chrono::high_resolution_clock::now();
    auto result2 = compiler.compile(graph);
    auto end2 = std::chrono::high_resolution_clock::now();
    
    std::cout << "Second compilation:\n";
    std::cout << "  From cache: " << (result2.from_cache ? "YES" : "no") << "\n";
    std::cout << "  Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() << " us\n";
    
    std::cout << "\n=== Execution Test ===\n";
    
    PlanExecutor executor(4);
    executor.set_verbose(false);
    
    float* input_buffer = new float[32 * 64];
    float* output_buffer = new float[32 * 128];
    
    for (int i = 0; i < 32 * 64; ++i) {
        input_buffer[i] = static_cast<float>(i) * 0.01f;
    }
    
    executor.execute(*result.plan, input_buffer, output_buffer);
    
    std::cout << "Execution completed.\n";
    
    delete[] input_buffer;
    delete[] output_buffer;
    
    std::cout << "\n=== All Tests Passed ===\n";
    
    return 0;
}
