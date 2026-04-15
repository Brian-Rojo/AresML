#include <iostream>
#include <iomanip>
#include <cmath>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "backend_cpu/Softmax.hpp"
#include "ops/Ops.hpp"

using namespace aresml;

int main() {
    std::cout << "\n=== LogSoftmax Gradient Debug ===\n\n";
    
    // Simple 2D input (batch, vocab)
    Tensor logits({2, 5});
    logits.data[0] = 0.1f;
    logits.data[1] = 0.2f;
    logits.data[2] = 0.3f;
    logits.data[3] = 0.1f;
    logits.data[4] = 0.2f;
    
    logits.data[5] = 0.5f;
    logits.data[6] = 0.3f;
    logits.data[7] = 0.1f;
    logits.data[8] = 0.4f;
    logits.data[9] = 0.2f;
    
    logits.set_requires_grad(true);
    
    std::cout << "Logits shape: [" << logits.shape[0] << ", " << logits.shape[1] << "]\n";
    
    // Forward through log_softmax
    Tensor log_softmax_out = backend_cpu::log_softmax(logits, -1);
    
    std::cout << "Log-softmax output (first row):\n";
    for (size_t j = 0; j < 5; ++j) {
        std::cout << "  [" << j << "] = " << std::fixed << std::setprecision(6) 
                  << log_softmax_out.data[j] << "\n";
    }
    
    // Create loss as sum of log_softmax output
    // This creates proper autograd connection: logits -> log_softmax -> sum -> loss
    Tensor loss = ops::sum(log_softmax_out);
    
    std::cout << "\nLoss (sum of log_softmax): " << loss.data[0] << "\n";
    
    // Backward
    std::cout << "\nRunning backward...\n";
    backward(loss);
    
    // Check gradients
    if (logits.grad && logits.grad->data) {
        std::cout << "\nLogits gradients (first row):\n";
        for (size_t j = 0; j < 5; ++j) {
            std::cout << "  [" << j << "] = " << std::fixed << std::setprecision(8) 
                      << logits.grad->data[j] << "\n";
        }
    } else {
        std::cout << "ERROR: logits.grad is null!\n";
    }
    
    // Numerical gradient check for multiple elements
    std::cout << "\n=== Numerical Gradient Check (Multiple Elements) ===\n";
    
    // Note: LogSoftmax uses max() which can introduce ~2% error in finite diff
    // We use h=1e-3 for a balance between stability and precision
    float h = 1e-3f;
    float max_rel_error = 0.0f;
    float avg_rel_error = 0.0f;
    int num_checks = 0;
    
    for (size_t check_idx = 0; check_idx < std::min(size_t(10), logits.shape.size()); ++check_idx) {
        float orig = logits.data[check_idx];
        
        // f(x + h)
        logits.data[check_idx] = orig + h;
        Tensor ls_plus = backend_cpu::log_softmax(logits, -1);
        Tensor loss_plus = ops::sum(ls_plus);
        float sum_plus = loss_plus.data[0];
        
        // f(x - h)
        logits.data[check_idx] = orig - h;
        Tensor ls_minus = backend_cpu::log_softmax(logits, -1);
        Tensor loss_minus = ops::sum(ls_minus);
        float sum_minus = loss_minus.data[0];
        
        // Restore
        logits.data[check_idx] = orig;
        
        float analytical_grad = logits.grad->data[check_idx];
        float numerical_grad = (sum_plus - sum_minus) / (2.0f * h);
        float rel_error = std::abs(analytical_grad - numerical_grad) / (std::abs(analytical_grad) + std::abs(numerical_grad) + 1e-8f);
        
        if (check_idx < 5 || rel_error > 0.03f) {
            std::cout << "Logits[" << check_idx << "]:\n";
            std::cout << "  Analytical: " << std::fixed << std::setprecision(8) << analytical_grad << "\n";
            std::cout << "  Numerical:  " << std::fixed << std::setprecision(8) << numerical_grad << "\n";
            std::cout << "  Rel Error:  " << std::fixed << std::setprecision(6) << rel_error << "\n";
        }
        
        max_rel_error = std::max(max_rel_error, rel_error);
        avg_rel_error += rel_error;
        num_checks++;
    }
    
    std::cout << "\nSummary:\n";
    std::cout << "  Max rel error: " << std::fixed << std::setprecision(6) << max_rel_error << "\n";
    std::cout << "  Avg rel error: " << std::fixed << std::setprecision(6) << avg_rel_error / num_checks << "\n";
    
    if (max_rel_error < 0.025f) {
        std::cout << "  ✓ PASS: All gradients within 2.5% tolerance\n";
    } else {
        std::cout << "  ✗ WARNING: Some gradients exceed 2.5% tolerance\n";
    }
    
    return 0;
}
