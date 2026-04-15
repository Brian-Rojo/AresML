#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/Linear.hpp"
#include "loss/CrossEntropy.hpp"
#include "loss/MSELoss.hpp"
#include "backend_cpu/Softmax.hpp"
#include "utils/GradientCheck.hpp"

using namespace aresml;

void test_linear_gradient() {
    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "TEST 1: Linear Layer Gradient Check\n";
    std::cout << std::string(50, '=') << "\n";
    
    nn::Linear linear(8, 4, false);
    
    Tensor input({2, 8});
    for (size_t i = 0; i < input.shape.size(); ++i) {
        input.data[i] = (static_cast<float>(i) - 4.0f) * 0.1f;
    }
    input.set_requires_grad(true);
    
    auto forward_fn = [&](Tensor& x) -> Tensor {
        return linear.forward(x);
    };
    
    try {
        auto result = utils::check_gradients(input, forward_fn, 1e-4f, 1e-3f);
        result.print();
        
        if (result.passed) {
            std::cout << "\n✓ Linear gradient check PASSED\n";
        } else {
            std::cout << "\n✗ Linear gradient check FAILED\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
    }
}

void test_logsoftmax_gradient() {
    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "TEST 2: LogSoftmax Gradient Check\n";
    std::cout << std::string(50, '=') << "\n";
    
    Tensor input({2, 10});
    for (size_t i = 0; i < input.shape.size(); ++i) {
        input.data[i] = (static_cast<float>(i) - 10.0f) * 0.2f;
    }
    input.set_requires_grad(true);
    
    auto forward_fn = [&](Tensor& x) -> Tensor {
        Tensor out = backend_cpu::log_softmax(x, -1);
        // Sum for scalar loss
        Tensor scalar({1});
        scalar.set_requires_grad(true);
        float sum = 0.0f;
        for (size_t i = 0; i < out.shape.size(); ++i) {
            sum += out.data[out.offset + i];
        }
        scalar.data[0] = sum;
        return scalar;
    };
    
    try {
        auto result = utils::check_gradients(input, forward_fn, 1e-4f, 1e-3f);
        result.print();
        
        if (result.passed) {
            std::cout << "\n✓ LogSoftmax gradient check PASSED\n";
        } else {
            std::cout << "\n✗ LogSoftmax gradient check FAILED\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
    }
}

void test_mse_gradient() {
    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "TEST 3: MSE Loss Gradient Check\n";
    std::cout << std::string(50, '=') << "\n";
    
    nn::Linear linear(6, 4, false);
    loss::MSELoss mse_loss;
    
    Tensor input({2, 6});
    for (size_t i = 0; i < input.shape.size(); ++i) {
        input.data[i] = (static_cast<float>(i) - 6.0f) * 0.15f;
    }
    input.set_requires_grad(true);
    
    Tensor target({2, 4});
    for (size_t i = 0; i < target.shape.size(); ++i) {
        target.data[i] = static_cast<float>(i) * 0.1f;
    }
    
    auto forward_fn = [&](Tensor& x) -> Tensor {
        return mse_loss.forward(linear.forward(x), target);
    };
    
    try {
        auto result = utils::check_gradients(input, forward_fn, 1e-4f, 1e-3f);
        result.print();
        
        if (result.passed) {
            std::cout << "\n✓ MSE gradient check PASSED\n";
        } else {
            std::cout << "\n✗ MSE gradient check FAILED\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
    }
}

void test_crossentropy_gradient_simple() {
    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "TEST 4: CrossEntropy Gradient Check (Simple)\n";
    std::cout << std::string(50, '=') << "\n";
    
    Tensor logits({2, 5});
    for (size_t i = 0; i < logits.shape.size(); ++i) {
        logits.data[i] = (static_cast<float>(i) - 5.0f) * 0.2f;
    }
    logits.set_requires_grad(true);
    
    Tensor targets({2});
    targets.data[0] = 1.0f;
    targets.data[1] = 3.0f;
    
    auto forward_fn = [&](Tensor& x) -> Tensor {
        loss::CrossEntropyLoss ce_loss(true);
        return ce_loss.forward(x, targets);
    };
    
    try {
        auto result = utils::check_gradients(logits, forward_fn, 1e-4f, 1e-3f);
        result.print();
        
        if (result.passed) {
            std::cout << "\n✓ CrossEntropy gradient check PASSED\n";
        } else {
            std::cout << "\n✗ CrossEntropy gradient check FAILED\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
    }
}

void test_stability_under_stress() {
    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "TEST 5: Stability Under Stress (100 iterations)\n";
    std::cout << std::string(50, '=') << "\n";
    
    nn::Linear linear(10, 5, false);
    loss::MSELoss mse_loss;
    
    Tensor input({4, 10});
    Tensor target({4, 5});
    
    for (size_t i = 0; i < input.shape.size(); ++i) {
        input.data[i] = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;
    }
    for (size_t i = 0; i < target.shape.size(); ++i) {
        target.data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
    input.set_requires_grad(true);
    
    int num_iterations = 100;
    int num_nan_detected = 0;
    int num_inf_detected = 0;
    
    for (int iter = 0; iter < num_iterations; ++iter) {
        try {
            input.grad->zero_();
            linear.weight.grad->zero_();
            linear.bias.grad->zero_();
            
            Tensor output = linear.forward(input);
            Tensor loss = mse_loss.forward(output, target);
            
            // Check for NaN/Inf in loss
            if (std::isnan(loss.data[loss.offset]) || std::isinf(loss.data[loss.offset])) {
                num_nan_detected++;
                if (iter < 3) {
                    std::cout << "Iteration " << iter << ": NaN/Inf detected in loss\n";
                }
            }
            
            backward(loss);
            
            // Check gradients
            bool has_nan = false;
            for (size_t i = 0; i < linear.weight.grad->shape.size(); ++i) {
                if (std::isnan(linear.weight.grad->data[i]) || std::isinf(linear.weight.grad->data[i])) {
                    has_nan = true;
                    num_nan_detected++;
                    break;
                }
            }
            
            if (iter % 25 == 0) {
                std::cout << "Iteration " << iter << ": loss=" << std::fixed << std::setprecision(6) 
                          << loss.data[loss.offset] << "\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Iteration " << iter << ": Exception - " << e.what() << "\n";
            break;
        }
    }
    
    std::cout << "\nCompleted " << num_iterations << " iterations\n";
    std::cout << "NaN/Inf detections: " << num_nan_detected << "\n";
    
    if (num_nan_detected == 0) {
        std::cout << "✓ Stress test PASSED\n";
    } else {
        std::cout << "✗ Stress test FAILED\n";
    }
}

int main() {
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         AresML v0.3 - Gradient Checking Test Suite              ║\n";
    std::cout << "║    Validating Backward Pass Correctness (FD-based)              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    set_debug(true);
    
    test_linear_gradient();
    test_logsoftmax_gradient();
    test_mse_gradient();
    test_crossentropy_gradient_simple();
    test_stability_under_stress();
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "All gradient check tests completed\n";
    std::cout << std::string(70, '=') << "\n";
    
    return 0;
}
