#include <iostream>
#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "core/GraphCompiler.hpp"
#include "nn/Linear.hpp"
#include "ops/Ops.hpp"

using namespace aresml;

int main() {
    std::cout << "=== Graph Compiler Test ===\n\n";
    
    std::cout << "1. Eager mode (no capture):\n";
    nn::Linear linear(4, 3, false);
    Tensor x({2, 4});
    x.set_requires_grad(true);
    x.is_leaf = true;
    
    for (int i = 0; i < 8; ++i) x.data[i] = (float)i * 0.1f;
    
    Tensor out = linear.forward(x);
    out.data[0] = out.sum();
    
    backward(out);
    
    std::cout << "   Forward done (eager)\n\n";
    
    std::cout << "2. Compiler mode with capture:\n";
    set_compiler_debug(true);
    set_compiler_mode(true);
    
    GraphRecorder::begin_capture();
    
    nn::Linear linear2(4, 3, false);
    Tensor x2({2, 4});
    x2.set_requires_grad(true);
    x2.is_leaf = true;
    
    for (int i = 0; i < 8; ++i) x2.data[i] = (float)i * 0.1f;
    
    Tensor out2 = linear2.forward(x2);
    std::cout << "   Forward with recording...\n";
    
    ComputationGraph* graph = GraphRecorder::end_capture();
    
    if (graph) {
        std::cout << "\n" << graph->to_string();
        std::cout << "Nodes captured: " << graph->node_count() << "\n";
        delete graph;
    }
    
    set_compiler_mode(false);
    set_compiler_debug(false);
    
    std::cout << "\n=== COMPILER TEST COMPLETED ===\n";
    return 0;
}