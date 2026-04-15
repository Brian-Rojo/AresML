#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/Linear.hpp"
#include "loss/MSELoss.hpp"
#include "loss/CrossEntropy.hpp"

using namespace aresml;

int main() {
    std::cout << "=== Debug CrossEntropy Backward ===\n\n";
    
    nn::Linear linear(4, 10, false);
    
    Tensor input({2, 4});
    for (size_t i = 0; i < input.shape.size(); ++i) {
        input.data[i] = static_cast<float>(i + 1) * 0.1f;
    }
    input.set_requires_grad(true);
    
    std::cout << "Input created\n";
    
    Tensor output = linear.forward(input);
    std::cout << "Forward done\n";
    
    Tensor target({2});
    target.data[0] = 3.0f;
    target.data[1] = 7.0f;
    
    loss::CrossEntropyLoss loss_fn(true);
    Tensor loss = loss_fn.forward(output, target);
    std::cout << "Loss: " << loss.data[loss.offset] << "\n";
    
    std::cout << "Calling backward...\n";
    set_debug(true);
    backward(loss);
    std::cout << "Backward done\n";
    
    if (linear.weight.grad && linear.weight.grad->data) {
        std::cout << "Grad exists, max: ";
        float max_g = 0.0f;
        for (size_t i = 0; i < linear.weight.grad->shape.size(); ++i) {
            max_g = std::max(max_g, std::abs(linear.weight.grad->data[i]));
        }
        std::cout << max_g << "\n";
    } else {
        std::cout << "Grad is null!\n";
    }
    
    std::cout << "Test complete\n";
    return 0;
}
