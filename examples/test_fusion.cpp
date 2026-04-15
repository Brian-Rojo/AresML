#include <iostream>

#include "core/IREngine.hpp"

using namespace aresml;

int main() {
    std::cout << "=== IR FUSION TEST ===\n\n";
    
    IRGraph graph;
    
    // Create a simple chain: matmul -> relu
    graph.add_node(IRType::MATMUL, {2, 4}, "mm1");
    graph.add_node(IRType::RELU, {2, 4}, "relu1");
    
    // Add matmul -> add (bias)
    graph.add_node(IRType::MATMUL, {2, 4}, "mm2");
    graph.add_node(IRType::ADD, {2, 4}, "bias2");
    
    // Add elementwise chain: add -> mul
    graph.add_node(IRType::ADD, {2, 4}, "add3");
    graph.add_node(IRType::MUL, {2, 4}, "mul3");
    
    std::cout << "Before fusion:\n";
    std::cout << graph.to_string() << "\n";
    
    // Run fusion pass
    FusionPass::run(graph);
    
    std::cout << "After fusion:\n";
    std::cout << graph.to_string() << "\n";
    
    // Count nodes
    size_t original = 6;
    size_t fused = graph.node_count();
    std::cout << "Nodes: " << original << " -> " << fused << " (fused " << (original - fused) << ")\n";
    
    std::cout << "\n=== DONE ===\n";
    
    return 0;
}