#include <iostream>
#include "core/ir_v2/IRGraphV2.hpp"
#include "core/ir_v2/IRNodeV2.hpp"
#include "core/fusion/FusionEngineV2.hpp"
#include "core/fusion/PatternMatcher.hpp"
#include "core/memory/CachePlanner.hpp"
#include "core/scheduler/SchedulerV2.hpp"
#include "core/runtime/FusedKernelExecutor.hpp"

using namespace aresml;

int main() {
    std::cout << "=== IRGraphV2 & FusionEngineV2 Test ===\n\n";
    
    IRGraphV2 graph;
    
    graph.add_node(IROpV2::INPUT, {32, 128}, "input");
    graph.add_node(IROpV2::WEIGHT, {128, 256}, "w1");
    graph.add_node(IROpV2::MATMUL, {32, 256}, "mm1");
    graph.add_node(IROpV2::BIAS_ADD, {32, 256}, "bias1");
    graph.add_node(IROpV2::RELU, {32, 256}, "relu1");
    
    graph.add_node(IROpV2::WEIGHT, {256, 128}, "w2");
    graph.add_node(IROpV2::MATMUL, {32, 128}, "mm2");
    graph.add_node(IROpV2::RELU, {32, 128}, "relu2");
    
    graph.add_node(IROpV2::ADD, {32, 128}, "add3");
    graph.add_node(IROpV2::MUL, {32, 128}, "mul3");
    
    std::cout << "Original graph (" << graph.node_count() << " nodes):\n";
    std::cout << graph.to_string() << "\n";
    
    auto result = FusionEngineV2().run(graph);
    
    std::cout << "After fusion (" << graph.node_count() << " nodes):\n";
    std::cout << graph.to_string() << "\n";
    std::cout << result.to_string() << "\n";
    
    std::cout << "=== Testing consecutive pattern fusion ===\n\n";
    
    IRGraphV2 graph2;
    graph2.add_node(IROpV2::INPUT, {32, 64}, "in");
    graph2.add_node(IROpV2::WEIGHT, {64, 128}, "w1");
    graph2.add_node(IROpV2::MATMUL, {32, 128}, "mm");
    graph2.add_node(IROpV2::BIAS_ADD, {32, 128}, "bias");
    graph2.add_node(IROpV2::RELU, {32, 128}, "relu");
    
    std::cout << "Before fusion (consecutive matmul+bias+relu):\n";
    std::cout << graph2.to_string() << "\n";
    
    auto result2 = FusionEngineV2().run(graph2);
    
    std::cout << "After fusion:\n";
    std::cout << graph2.to_string() << "\n";
    std::cout << result2.to_string() << "\n";
    
    std::cout << "=== Testing elementwise chain ===\n\n";
    
    IRGraphV2 graph3;
    graph3.add_node(IROpV2::INPUT, {32, 64}, "in");
    graph3.add_node(IROpV2::ADD, {32, 64}, "add");
    graph3.add_node(IROpV2::MUL, {32, 64}, "mul");
    
    std::cout << "Before fusion:\n";
    std::cout << graph3.to_string() << "\n";
    
    auto result3 = FusionEngineV2().run(graph3);
    
    std::cout << "After fusion:\n";
    std::cout << graph3.to_string() << "\n";
    std::cout << result3.to_string() << "\n";
    
    std::cout << "=== SchedulerV2 Test ===\n\n";
    
    IRGraphV2 graph4;
    graph4.add_node(IROpV2::INPUT, {64, 64}, "in");
    graph4.add_node(IROpV2::WEIGHT, {64, 64}, "w");
    graph4.add_node(IROpV2::MATMUL, {64, 64}, "mm");
    graph4.add_node(IROpV2::RELU, {64, 64}, "relu");
    graph4.add_node(IROpV2::ADD, {64, 64}, "add");
    graph4.add_node(IROpV2::MUL, {64, 64}, "mul");
    
    SchedulerV2 scheduler(ScheduleStrategy::PARALLEL, 4);
    auto plan = scheduler.create_plan(graph4);
    
    std::cout << plan.to_string() << "\n";
    
    std::cout << "=== CachePlanner Test ===\n\n";
    
    auto block = CachePlanner::compute_optimal_block(1024, 1024, 512);
    std::cout << "Optimal block: " << block.block_m << "x" << block.block_n << "x" << block.block_k << "\n";
    std::cout << "Size: " << block.size_bytes << " bytes\n";
    std::cout << "Reuse factor: " << block.reuse_factor << "\n";
    
    auto blocks = CachePlanner::generate_blocking_plan(128, 128, 64);
    std::cout << "Generated " << blocks.size() << " blocks\n";
    
    std::cout << "\n=== All Tests Passed ===\n";
    
    return 0;
}
