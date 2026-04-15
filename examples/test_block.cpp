#include <iostream>
#include <vector>
#include <cmath>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/RMSNorm.hpp"
#include "nn/Attention.hpp"
#include "nn/FFN.hpp"
#include "nn/TransformerBlock.hpp"

using namespace aresml;

int main() {
    std::cout << "=== Test TransformerBlock ===\n";
    
    try {
        nn::TransformerBlock block(32, 4);
        std::cout << "Block created\n";
        
        Tensor input({2, 4, 32});
        input.fill(0.1f);
        input.set_requires_grad(true);
        
        std::cout << "Input shape: " << input.shape[0] << "x" << input.shape[1] << "x" << input.shape[2] << "\n";
        
        std::cout << "Calling forward...\n";
        Tensor output = block.forward(input);
        std::cout << "Output shape: " << output.shape[0] << "x" << output.shape[1] << "x" << output.shape[2] << "\n";
        
        std::cout << "Done!\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
