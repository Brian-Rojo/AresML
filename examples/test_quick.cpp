#include <iostream>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/Linear.hpp"
#include "loss/MSELoss.hpp"

using namespace aresml;

int main() {
    std::cout << "=== Quick Test ===\n\n";
    
    nn::Linear linear(2, 1, false);
    
    Tensor input({1, 2});
    input.data[0] = 0.5f;
    input.data[1] = 0.3f;
    input.set_requires_grad(true);
    
    Tensor output = linear.forward(input);
    
    Tensor target({1});
    target.data[0] = 0.8f;
    
    loss::MSELoss mse;
    Tensor loss = mse.forward(output, target);
    
    std::cout << "Loss: " << loss.data[0] << "\n";
    
    backward(loss);
    
    if (linear.weight.grad && linear.weight.grad->data) {
        bool nonzero = false;
        for (size_t i = 0; i < linear.weight.shape.size(); ++i) {
            if (std::abs(linear.weight.grad->data[i]) > 1e-6f) {
                nonzero = true;
                break;
            }
        }
        
        if (nonzero) {
            std::cout << "SUCCESS! grad: ";
            for (size_t i = 0; i < linear.weight.shape.size(); ++i) {
                std::cout << linear.weight.grad->data[i] << " ";
            }
            std::cout << "\n";
        } else {
            std::cout << "FAILED: grad is zero!\n";
        }
    } else {
        std::cout << "FAILED: no grad\n";
    }
    
    return 0;
}
