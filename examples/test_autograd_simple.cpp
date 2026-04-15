#include "../core/Tensor.hpp"
#include "../core/Autograd.hpp"
#include "../nn/Linear.hpp"
#include "../loss/MSELoss.hpp"
#include <iostream>

using namespace aresml;

int main() {
    std::cout << "=== Simple Autograd Test ===" << std::endl;
    
    // Simple linear layer
    nn::Linear linear("test_linear", 4, 2);
    
    // Input
    Tensor input(Shape{2, 4}, false);
    input.set_requires_grad(true);
    input.is_leaf = true;
    get_engine().register_leaf(&input);
    
    for (int i = 0; i < 8; ++i) input.data[i] = (float)(i % 4);
    
    std::cout << "Input: " << input.shape[0] << "x" << input.shape[1] << std::endl;
    std::cout << "Linear weight: " << linear.weight.shape[0] << "x" << linear.weight.shape[1] << std::endl;
    
    get_engine().zero_grad();
    
    // Forward
    Tensor output = linear.forward(input);
    std::cout << "Output shape: " << output.shape[0] << "x" << output.shape[1] << std::endl;
    std::cout << "Output requires_grad: " << output.requires_grad << std::endl;
    
    // Target
    Tensor target(Shape{2, 2}, false);
    for (int i = 0; i < 4; ++i) target.data[i] = 0.0f;
    
    // Simple MSE loss
    loss::MSELoss mse;
    Tensor loss = mse.forward(output, target);
    
    std::cout << "Loss: " << loss.data[loss.offset] << std::endl;
    std::cout << "Loss requires_grad: " << loss.requires_grad << std::endl;
    
    get_engine().backward(&loss);
    
    std::cout << "\n=== Gradient Check ===" << std::endl;
    
    if (input.grad && input.grad->data) {
        float max_in = 0;
        for (size_t i = 0; i < input.grad->shape.size(); ++i) {
            max_in = std::max(max_in, std::abs(input.grad->data[input.grad->offset + i]));
        }
        std::cout << "input.grad max: " << max_in << std::endl;
    } else {
        std::cout << "input.grad: null" << std::endl;
    }
    
    if (linear.weight.grad && linear.weight.grad->data) {
        float max_w = 0;
        for (size_t i = 0; i < linear.weight.grad->shape.size(); ++i) {
            max_w = std::max(max_w, std::abs(linear.weight.grad->data[linear.weight.grad->offset + i]));
        }
        std::cout << "weight.grad max: " << max_w << std::endl;
    } else {
        std::cout << "weight.grad: null" << std::endl;
    }
    
    std::cout << "=== Test Complete ===" << std::endl;
    return 0;
}