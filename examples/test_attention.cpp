#include <iostream>
#include <vector>
#include <cmath>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/Attention.hpp"

using namespace aresml;

int main() {
    std::cout << "=== Test Attention ===\n";
    
    try {
        nn::Attention attn(32, 4);
        std::cout << "Attention created\n";
        
        Tensor input({1, 4, 32});
        input.fill(0.1f);
        input.requires_grad = true;
        
        std::cout << "Calling forward...\n";
        Tensor output = attn.forward(input);
        std::cout << "Output shape: " << output.shape.n << "D\n";
        
        std::cout << "Done!\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
