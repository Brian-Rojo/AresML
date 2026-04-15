#pragma once

#include "../core/Tensor.hpp"
#include "../core/Autograd.hpp"
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

namespace aresml {
namespace utils {

struct GradientCheckResult {
    bool passed;
    float max_error;
    float mean_error;
    int num_checked;
    int num_failed;
    
    void print() const {
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "Gradient Check: " << (passed ? "PASSED" : "FAILED") << "\n";
        std::cout << "  Max error: " << max_error << "\n";
        std::cout << "  Mean error: " << mean_error << "\n";
        std::cout << "  Elements checked: " << num_checked << "\n";
        std::cout << "  Elements with error > 1e-5: " << num_failed << "\n";
    }
};

// Forward difference approximation: f'(x) ≈ (f(x+h) - f(x)) / h
// Central difference approximation: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
// We use central difference for better accuracy

inline GradientCheckResult check_gradients(
    Tensor& input,
    std::function<Tensor(Tensor&)> forward_fn,
    float epsilon = 1e-4f,
    float tolerance = 1e-4f
) {
    GradientCheckResult result{true, 0.0f, 0.0f, 0, 0};
    
    if (!input.requires_grad || !input.data) {
        std::cerr << "Input tensor must require grad and have data\n";
        result.passed = false;
        return result;
    }
    
    // Step 1: Run forward+backward to get analytical gradients
    Tensor output = forward_fn(input);
    
    if (!output.requires_grad) {
        std::cerr << "Output tensor must require grad\n";
        result.passed = false;
        return result;
    }
    
    backward(output);
    
    if (!input.grad || !input.grad->data) {
        std::cerr << "Input gradient not computed\n";
        result.passed = false;
        return result;
    }
    
    // Store analytical gradients
    std::vector<float> analytical_grads(input.shape.size());
    for (size_t i = 0; i < input.shape.size(); ++i) {
        analytical_grads[i] = input.grad->data[input.grad->offset + i];
    }
    
    // Zero out gradients for numerical check
    if (input.grad) {
        input.grad->zero_();
    }
    
    // Step 2: Compute numerical gradients via finite differences
    std::vector<float> numerical_grads(input.shape.size());
    
    for (size_t i = 0; i < input.shape.size(); ++i) {
        float original = input.data[input.offset + i];
        
        // Clear any accumulated gradients
        zero_grad();
        
        // f(x + epsilon)
        input.data[input.offset + i] = original + epsilon;
        Tensor output_plus = forward_fn(input);
        float loss_plus = output_plus.data[output_plus.offset];
        
        // Clear gradients again
        zero_grad();
        
        // f(x - epsilon)
        input.data[input.offset + i] = original - epsilon;
        Tensor output_minus = forward_fn(input);
        float loss_minus = output_minus.data[output_minus.offset];
        
        // Restore original
        input.data[input.offset + i] = original;
        
        // Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
        numerical_grads[i] = (loss_plus - loss_minus) / (2.0f * epsilon);
    }
    
    // Step 3: Compare analytical vs numerical
    float total_error = 0.0f;
    result.num_checked = input.shape.size();
    
    for (size_t i = 0; i < input.shape.size(); ++i) {
        float analytical = analytical_grads[i];
        float numerical = numerical_grads[i];
        
        // Compute relative error
        float diff = std::abs(analytical - numerical);
        float denom = std::abs(analytical) + std::abs(numerical) + 1e-8f;
        float relative_error = diff / denom;
        
        total_error += relative_error;
        result.max_error = std::max(result.max_error, relative_error);
        
        if (relative_error > tolerance) {
            result.num_failed++;
            if (result.num_failed <= 5) {  // Print first 5 failures
                std::cout << "  Index " << i << ":\n";
                std::cout << "    analytical=" << std::fixed << std::setprecision(8) << analytical << "\n";
                std::cout << "    numerical=" << std::fixed << std::setprecision(8) << numerical << "\n";
                std::cout << "    rel_error=" << std::fixed << std::setprecision(8) << relative_error << "\n";
            }
        }
    }
    
    result.mean_error = total_error / input.shape.size();
    result.passed = result.num_failed == 0;
    
    return result;
}

// Batch version for testing multiple inputs
inline void check_gradients_batch(
    std::vector<Tensor*> inputs,
    std::function<Tensor()> forward_fn,
    float epsilon = 1e-4f,
    float tolerance = 1e-4f
) {
    std::cout << "\n=== BATCH GRADIENT CHECK ===\n";
    
    int total_passed = 0;
    int total_failed = 0;
    
    for (size_t idx = 0; idx < inputs.size(); ++idx) {
        Tensor* input = inputs[idx];
        
        if (!input || !input->requires_grad) {
            std::cout << "Input " << idx << ": SKIPPED (no require_grad)\n";
            continue;
        }
        
        std::cout << "\nInput " << idx << " (shape=" << input->shape.size() << "):\n";
        
        // Create a closure that captures the forward function
        auto local_forward = [&](Tensor& dummy) -> Tensor {
            return forward_fn();
        };
        
        auto result = check_gradients(*input, local_forward, epsilon, tolerance);
        result.print();
        
        if (result.passed) total_passed++;
        else total_failed++;
    }
    
    std::cout << "\n=== BATCH SUMMARY ===\n";
    std::cout << "Passed: " << total_passed << " / Failed: " << total_failed << "\n";
}

}  // namespace utils
}  // namespace aresml
