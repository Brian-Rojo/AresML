#include <iostream>
#include <iomanip>
#include <cmath>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/Linear.hpp"
#include "loss/MSELoss.hpp"

using namespace aresml;

int main() {
    std::cout << "\n=== Minimal Gradient Debug: Linear + MSE ===\n\n";
    
    // Simple setup
    nn::Linear linear(2, 3, false);
    
    // Set weights to known values for debugging
    linear.weight.data[0] = 0.1f;
    linear.weight.data[1] = 0.2f;
    linear.weight.data[2] = 0.3f;
    linear.weight.data[3] = 0.4f;
    linear.weight.data[4] = 0.5f;
    linear.weight.data[5] = 0.6f;
    
    Tensor input({1, 2});  // Batch size 1
    input.data[0] = 1.0f;
    input.data[1] = 2.0f;
    input.set_requires_grad(true);
    
    Tensor target({1, 3});
    target.data[0] = 0.5f;
    target.data[1] = 0.3f;
    target.data[2] = 0.8f;
    
    std::cout << "Input: [" << input.data[0] << ", " << input.data[1] << "]\n";
    std::cout << "Target: [" << target.data[0] << ", " << target.data[1] << ", " 
              << target.data[2] << "]\n";
    std::cout << "Weight shape: " << linear.weight.shape[0] << " x " << linear.weight.shape[1] << "\n\n";
    
    // Forward
    Tensor output = linear.forward(input);
    std::cout << "Output: [" << std::fixed << std::setprecision(6) 
              << output.data[0] << ", " << output.data[1] << ", " << output.data[2] << "]\n";
    
    // Compute loss
    loss::MSELoss loss_fn;
    Tensor loss = loss_fn.forward(output, target);
    std::cout << "Loss: " << loss.data[0] << "\n\n";
    
    // Backward
    std::cout << "Running backward...\n";
    backward(loss);
    
    // Check if gradients computed
    if (input.grad && input.grad->data) {
        std::cout << "\nInput gradients:\n";
        for (size_t i = 0; i < input.shape.size(); ++i) {
            std::cout << "  [" << i << "] = " << std::fixed << std::setprecision(8) 
                      << input.grad->data[input.grad->offset + i] << "\n";
        }
    }
    
    if (linear.weight.grad && linear.weight.grad->data) {
        std::cout << "\nWeight gradients (first 6):\n";
        for (size_t i = 0; i < std::min(size_t(6), linear.weight.grad->shape.size()); ++i) {
            std::cout << "  [" << i << "] = " << std::fixed << std::setprecision(8) 
                      << linear.weight.grad->data[i] << "\n";
        }
    }
    
    // Manual numerical gradient check for input[0]
    std::cout << "\n=== Manual Numerical Gradient Check ===\n";
    float h = 1e-4f;
    float orig = input.data[0];
    
    // f(x + h)
    input.data[0] = orig + h;
    Tensor out_plus = linear.forward(input);
    Tensor loss_plus = loss_fn.forward(out_plus, target);
    float f_plus = loss_plus.data[0];
    
    // f(x - h)
    input.data[0] = orig - h;
    Tensor out_minus = linear.forward(input);
    Tensor loss_minus = loss_fn.forward(out_minus, target);
    float f_minus = loss_minus.data[0];
    
    // Restore
    input.data[0] = orig;
    
    float numerical_grad = (f_plus - f_minus) / (2.0f * h);
    float analytical_grad = input.grad->data[input.grad->offset];
    
    std::cout << "Input[0] gradient:\n";
    std::cout << "  Analytical: " << std::fixed << std::setprecision(8) << analytical_grad << "\n";
    std::cout << "  Numerical:  " << std::fixed << std::setprecision(8) << numerical_grad << "\n";
    std::cout << "  Difference: " << std::fixed << std::setprecision(8) 
              << std::abs(analytical_grad - numerical_grad) << "\n";
    std::cout << "  Rel Error:  " << std::fixed << std::setprecision(6) 
              << std::abs(analytical_grad - numerical_grad) / (std::abs(analytical_grad) + std::abs(numerical_grad) + 1e-8f) << "\n";
    
    return 0;
}
