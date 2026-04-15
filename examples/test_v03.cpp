#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/Linear.hpp"
#include "loss/MSELoss.hpp"
#include "loss/CrossEntropy.hpp"
#include "optim/SGD.hpp"

using namespace aresml;

int main() {
    std::cout << "=== AresML v0.3 Test Suite ===\n\n";
    
    int passed = 0;
    int total = 5;
    
    {
        std::cout << "TEST 1: Linear + MSE Training (100 steps)\n";
        
        nn::Linear linear(4, 3, false);
        
        float initial_loss = 0.0f;
        bool has_nan = false;
        
        for (int step = 0; step < 100; ++step) {
            Tensor input({2, 4});
            input.data[0] = static_cast<float>(rand()) / RAND_MAX;
            input.data[1] = static_cast<float>(rand()) / RAND_MAX;
            input.data[2] = static_cast<float>(rand()) / RAND_MAX;
            input.data[3] = static_cast<float>(rand()) / RAND_MAX;
            input.set_requires_grad(true);
            
            Tensor target({2, 3});
            target.fill(0.5f);
            
            Tensor output = linear.forward(input);
            loss::MSELoss mse;
            Tensor loss = mse.forward(output, target);
            
            if (std::isnan(loss.data[loss.offset])) {
                has_nan = true;
                break;
            }
            
            if (step == 0) initial_loss = loss.data[loss.offset];
            
            backward(loss);
            
            float lr = 0.01f;
            if (linear.weight.grad && linear.weight.grad->data) {
                for (size_t j = 0; j < linear.weight.shape.size(); ++j) {
                    linear.weight.data[j] -= lr * linear.weight.grad->data[j];
                }
            }
            
            get_engine().clear();
        }
        
        if (!has_nan) {
            std::cout << "  PASSED: 100 steps completed\n";
            passed++;
        } else {
            std::cout << "  FAILED: NaN detected\n";
        }
    }
    
    {
        std::cout << "\nTEST 2: 3-Layer MLP Gradient Flow\n";
        
        nn::Linear linear1(4, 8, false);
        nn::Linear linear2(8, 8, false);
        nn::Linear linear3(8, 3, false);
        
        Tensor input({2, 4});
        input.set_requires_grad(true);
        
        Tensor out1 = linear1.forward(input);
        Tensor out2 = linear2.forward(out1);
        Tensor out3 = linear3.forward(out2);
        
        Tensor target({2, 3});
        target.fill(0.5f);
        
        loss::MSELoss mse;
        Tensor loss = mse.forward(out3, target);
        
        backward(loss);
        
        bool grad_ok = true;
        if (!linear1.weight.grad || !linear1.weight.grad->data) grad_ok = false;
        if (!linear2.weight.grad || !linear2.weight.grad->data) grad_ok = false;
        if (!linear3.weight.grad || !linear3.weight.grad->data) grad_ok = false;
        
        if (grad_ok) {
            std::cout << "  PASSED: gradients flow through 3 layers\n";
            passed++;
        } else {
            std::cout << "  FAILED: gradient flow broken\n";
        }
        
        get_engine().clear();
    }
    
    {
        std::cout << "\nTEST 3: CrossEntropy Forward (no NaN)\n";
        
        nn::Linear linear(4, 10, false);
        
        Tensor input({2, 4});
        input.fill(0.5f);
        
        Tensor output = linear.forward(input);
        
        Tensor target({2});
        target.data[0] = 3.0f;
        target.data[1] = 7.0f;
        
        loss::CrossEntropyLoss loss_fn(true);
        Tensor loss = loss_fn.forward(output, target);
        
        if (!std::isnan(loss.data[loss.offset]) && !std::isinf(loss.data[loss.offset])) {
            std::cout << "  PASSED: loss = " << loss.data[loss.offset] << "\n";
            passed++;
        } else {
            std::cout << "  FAILED: NaN/Inf in loss\n";
        }
        
        get_engine().clear();
    }
    
    {
        std::cout << "\nTEST 4: SGD Optimizer Step\n";
        
        nn::Linear linear(2, 1, false);
        float w_before = linear.weight.data[0];
        
        Tensor input({1, 2});
        input.data[0] = 0.5f;
        input.data[1] = 0.3f;
        input.set_requires_grad(true);
        
        Tensor target({1});
        target.data[0] = 0.8f;
        
        std::vector<Tensor*> params = {&linear.weight};
        optim::SGD sgd(params, 0.1f);
        
        for (int i = 0; i < 10; ++i) {
            get_engine().zero_grad();
            Tensor output = linear.forward(input);
            loss::MSELoss mse;
            Tensor loss = mse.forward(output, target);
            backward(loss);
            sgd.step();
            get_engine().clear();
        }
        
        float w_after = linear.weight.data[0];
        if (std::abs(w_after - w_before) > 1e-4f) {
            std::cout << "  PASSED: SGD updated weights\n";
            passed++;
        } else {
            std::cout << "  FAILED: weights unchanged\n";
        }
    }
    
    {
        std::cout << "\nTEST 5: Long Training (500 steps)\n";
        
        nn::Linear linear(2, 1, false);
        
        bool has_nan = false;
        float last_loss = 0.0f;
        
        for (int step = 0; step < 500; ++step) {
            Tensor input({1, 2});
            input.data[0] = static_cast<float>(rand()) / RAND_MAX;
            input.data[1] = static_cast<float>(rand()) / RAND_MAX;
            input.set_requires_grad(true);
            
            Tensor target({1});
            target.data[0] = input.data[0] * 2.0f + 0.1f;
            
            Tensor output = linear.forward(input);
            loss::MSELoss mse;
            Tensor loss = mse.forward(output, target);
            
            last_loss = loss.data[loss.offset];
            if (std::isnan(last_loss)) {
                has_nan = true;
                break;
            }
            
            backward(loss);
            
            float lr = 0.01f;
            if (linear.weight.grad && linear.weight.grad->data) {
                for (size_t j = 0; j < linear.weight.shape.size(); ++j) {
                    linear.weight.data[j] -= lr * linear.weight.grad->data[j];
                }
            }
            
            get_engine().clear();
        }
        
        if (!has_nan) {
            std::cout << "  PASSED: 500 steps, final loss = " << last_loss << "\n";
            passed++;
        } else {
            std::cout << "  FAILED: NaN detected\n";
        }
    }
    
    std::cout << "\n=== SUMMARY ===\n";
    std::cout << "Passed: " << passed << "/" << total << "\n";
    
    if (passed == total) {
        std::cout << "All tests PASSED! AresML v0.3 stable.\n";
        return 0;
    }
    return 1;
}
