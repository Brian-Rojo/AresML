#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/RMSNorm.hpp"
#include "nn/Linear.hpp"
#include "optim/Adam.hpp"
#include "loss/CrossEntropy.hpp"

using namespace aresml;

int main() {
    std::cout << "=== Test RMSNorm ===\n";
    
    try {
        nn::RMSNorm norm(32);
        std::cout << "RMSNorm created\n";
        
        Tensor input({2, 16, 32});
        input.fill(0.5f);
        input.requires_grad = true;
        
        std::cout << "Input shape: " << input.shape[0] << "x" << input.shape[1] << "x" << input.shape[2] << "\n";
        
        Tensor output = norm.forward(input);
        std::cout << "Output shape: " << output.shape[0] << "x" << output.shape[1] << "x" << output.shape[2] << "\n";
        
        std::cout << "Done!\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
