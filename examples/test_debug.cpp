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
    set_debug(true);
    std::cout << "=== DEBUG MODE: Linear + MSE ===\n\n";
    
    nn::Linear linear(4, 3, false);
    std::cout << "linear.weight: " << &linear.weight << "\n";
    std::cout << "linear.weight.is_leaf: " << linear.weight.is_leaf << "\n";
    
    Tensor input({2, 4});
    for (size_t i = 0; i < input.shape.size(); ++i) {
        input.data[i] = static_cast<float>(i + 1) * 0.1f;
    }
    input.set_requires_grad(true);
    
    std::cout << "input: " << &input << "\n";
    std::cout << "input.is_leaf: " << input.is_leaf << "\n";
    
    Tensor output = linear.forward(input);
    std::cout << "output: " << &output << "\n";
    std::cout << "output.op: " << (output.op ? "exists" : "nullptr") << "\n";
    std::cout << "output.inputs.size: " << output.inputs.size() << "\n";
    if (!output.inputs.empty()) {
        std::cout << "output.inputs[0]: " << output.inputs[0] << "\n";
    }
    
    Tensor target({2, 3});
    for (size_t i = 0; i < target.shape.size(); ++i) {
        target.data[i] = 0.5f;
    }
    
    loss::MSELoss mse;
    Tensor loss = mse.forward(output, target);
    
    std::cout << "loss: " << &loss << "\n";
    std::cout << "loss.op: " << (loss.op ? "exists" : "nullptr") << "\n";
    std::cout << "loss.inputs.size: " << loss.inputs.size() << "\n";
    
    std::cout << "\n=== Calling backward ===\n";
    backward(loss);
    
    std::cout << "\n=== After backward ===\n";
    std::cout << "linear.weight.grad: " << linear.weight.grad << "\n";
    if (linear.weight.grad && linear.weight.grad->data) {
        std::cout << "weight.grad[0:4]: ";
        for (size_t i = 0; i < std::min(size_t(4), linear.weight.shape.size()); ++i) {
            std::cout << linear.weight.grad->data[i] << " ";
        }
        std::cout << "\n";
    }
    
    return 0;
}
