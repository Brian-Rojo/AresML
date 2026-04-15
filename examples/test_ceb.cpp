#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/Linear.hpp"
#include "loss/CrossEntropy.hpp"

using namespace aresml;

int main() {
    std::cout << "=== Test CrossEntropy Backward ===\n\n";
    
    nn::Linear linear(4, 10, false);
    
    Tensor input({1, 4});
    input.data[0] = 0.1f;
    input.data[1] = 0.2f;
    input.data[2] = 0.3f;
    input.data[3] = 0.4f;
    input.set_requires_grad(true);
    
    Tensor output = linear.forward(input);
    
    Tensor target({1});
    target.data[0] = 3.0f;
    
    loss::CrossEntropyLoss loss_fn(true);
    Tensor loss = loss_fn.forward(output, target);
    
    std::cout << "Loss: " << loss.data[loss.offset] << "\n";
    
    std::cout << "Calling backward...\n";
    backward(loss);
    std::cout << "Backward done\n";
    
    if (linear.weight.grad && linear.weight.grad->data) {
        float max_g = 0.0f;
        for (size_t i = 0; i < linear.weight.grad->shape.size(); ++i) {
            max_g = std::max(max_g, std::abs(linear.weight.grad->data[i]));
        }
        std::cout << "Grad max: " << max_g << "\n";
        if (max_g > 1e-10f) {
            std::cout << "PASSED\n";
        } else {
            std::cout << "FAILED: grad is zero\n";
        }
    } else {
        std::cout << "FAILED: grad is null\n";
    }
    
    return 0;
}
