#include <iostream>
#include <iomanip>
#include <cmath>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "loss/CrossEntropy.hpp"
#include "nn/Linear.hpp"
#include "ops/Ops.hpp"

using namespace aresml;

int main() {
    std::cout << "\n=== CrossEntropy Gradient Debug ===\n\n";
    
    // Create a simple linear layer + cross entropy
    size_t batch_size = 2;
    size_t input_size = 3;
    size_t num_classes = 4;
    
    Tensor input({batch_size, input_size});
    input.data[0] = 0.1f; input.data[1] = 0.2f; input.data[2] = 0.3f;
    input.data[3] = 0.4f; input.data[4] = 0.5f; input.data[5] = 0.6f;
    input.set_requires_grad(true);
    
    nn::Linear linear(input_size, num_classes);
    Tensor logits = linear.forward(input);
    
    std::cout << "Logits shape: [" << logits.shape[0] << ", " << logits.shape[1] << "]\n";
    std::cout << "Logits (first row): ";
    for (size_t j = 0; j < num_classes; ++j) {
        std::cout << std::fixed << std::setprecision(4) << logits.data[j] << " ";
    }
    std::cout << "\n\n";
    
    // Create targets: class [1, 2]
    Tensor targets({batch_size});
    targets.data[0] = 1.0f;
    targets.data[1] = 2.0f;
    
    // Forward
    loss::CrossEntropyLoss loss_fn;
    Tensor loss = loss_fn.forward(logits, targets);
    
    std::cout << "Loss: " << std::fixed << std::setprecision(6) << loss.data[0] << "\n";
    
    // Backward
    std::cout << "\nRunning backward...\n";
    backward(loss);
    
    // Check gradients
    if (logits.grad && logits.grad->data) {
        std::cout << "\nLogits gradients (first row):\n";
        for (size_t j = 0; j < num_classes; ++j) {
            std::cout << "  [" << j << "] = " << std::fixed << std::setprecision(8) 
                      << logits.grad->data[j] << "\n";
        }
    } else {
        std::cout << "ERROR: logits.grad is null!\n";
    }
    
    if (input.grad && input.grad->data) {
        std::cout << "\nInput gradients (first row):\n";
        for (size_t j = 0; j < input_size; ++j) {
            std::cout << "  [" << j << "] = " << std::fixed << std::setprecision(8) 
                      << input.grad->data[j] << "\n";
        }
    }
    
    // Numerical gradient check
    std::cout << "\n=== Numerical Gradient Check (Logits[0,1]) ===\n";
    
    // Save original logits values
    std::vector<float> logits_orig(logits.shape.size());
    for (size_t i = 0; i < logits.shape.size(); ++i) {
        logits_orig[i] = logits.data[i];
    }
    
    float h = 1e-3f;
    
    // f(x + h) - perturb logits directly
    logits.data[1] = logits_orig[1] + h;
    Tensor loss_plus = loss_fn.forward(logits, targets);
    float loss_val_plus = loss_plus.data[0];
    
    // f(x - h) - perturb logits directly
    logits.data[1] = logits_orig[1] - h;
    Tensor loss_minus = loss_fn.forward(logits, targets);
    float loss_val_minus = loss_minus.data[0];
    
    // Restore
    logits.data[1] = logits_orig[1];
    
    float numerical_grad = (loss_val_plus - loss_val_minus) / (2.0f * h);
    float analytical_grad = logits.grad->data[1];
    float rel_error = std::abs(analytical_grad - numerical_grad) / (std::abs(analytical_grad) + std::abs(numerical_grad) + 1e-8f);
    
    std::cout << "Logits[0,1] gradient:\n";
    std::cout << "  Analytical: " << std::fixed << std::setprecision(8) << analytical_grad << "\n";
    std::cout << "  Numerical:  " << std::fixed << std::setprecision(8) << numerical_grad << "\n";
    std::cout << "  Rel Error:  " << std::fixed << std::setprecision(6) << rel_error << "\n";
    
    if (rel_error < 0.025f) {
        std::cout << "  ✓ PASS: Gradient within 2.5% tolerance\n";
    } else {
        std::cout << "  ✗ FAIL: Gradient error exceeds tolerance\n";
    }
    
    return 0;
}
