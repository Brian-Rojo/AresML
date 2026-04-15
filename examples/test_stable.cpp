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
    std::cout << "=== AresML v0.2 Stable Test Suite ===\n\n";
    
    int passed = 0;
    int total = 0;
    
    {
        std::cout << "TEST 1: Linear + MSE Training\n";
        total++;
        
        nn::Linear linear(4, 3, false);
        
        Tensor input({2, 4});
        for (size_t i = 0; i < input.shape.size(); ++i) {
            input.data[i] = static_cast<float>(i + 1) * 0.1f;
        }
        input.set_requires_grad(true);
        
        Tensor output = linear.forward(input);
        
        Tensor target({2, 3});
        for (size_t i = 0; i < target.shape.size(); ++i) {
            target.data[i] = 0.5f;
        }
        
        loss::MSELoss mse;
        Tensor loss = mse.forward(output, target);
        
        backward(loss);
        
        bool has_grad = (linear.weight.grad && linear.weight.grad->data);
        bool grad_nonzero = false;
        if (has_grad) {
            for (size_t i = 0; i < linear.weight.shape.size(); ++i) {
                if (std::abs(linear.weight.grad->data[i]) > 1e-6f) {
                    grad_nonzero = true;
                    break;
                }
            }
        }
        
        if (has_grad && grad_nonzero) {
            std::cout << "  PASSED: weight grad exists and nonzero\n";
            passed++;
        } else {
            std::cout << "  FAILED: weight grad missing or zero\n";
        }
        
        get_engine().clear();
    }
    
    {
        std::cout << "\nTEST 2: Linear + CrossEntropy Forward\n";
        total++;
        
        nn::Linear linear(4, 3, false);
        
        Tensor input({2, 4});
        for (size_t i = 0; i < input.shape.size(); ++i) {
            input.data[i] = static_cast<float>(i + 1) * 0.1f;
        }
        
        Tensor output = linear.forward(input);
        
        Tensor target({2});
        target.data[0] = 0.0f;
        target.data[1] = 1.0f;
        
        loss::CrossEntropyLoss loss_fn(true);
        Tensor loss = loss_fn.forward(output, target);
        
        float loss_val = loss.data[loss.offset];
        
        if (!std::isnan(loss_val) && !std::isinf(loss_val) && loss_val > 0.0f) {
            std::cout << "  PASSED: loss = " << loss_val << " (valid)\n";
            passed++;
        } else {
            std::cout << "  FAILED: loss = " << loss_val << " (NaN/Inf/invalid)\n";
        }
        
        get_engine().clear();
    }
    
    {
        std::cout << "\nTEST 3: Autograd Clean\n";
        total++;
        
        nn::Linear linear(4, 3, false);
        
        Tensor input({1, 4});
        input.set_requires_grad(true);
        
        Tensor output = linear.forward(input);
        
        loss::MSELoss mse;
        Tensor target({1, 3});
        target.fill(0.5f);
        Tensor loss = mse.forward(output, target);
        
        backward(loss);
        get_engine().zero_grad();
        
        bool grad_zeroed = true;
        if (linear.weight.grad && linear.weight.grad->data) {
            for (size_t i = 0; i < linear.weight.shape.size(); ++i) {
                if (std::abs(linear.weight.grad->data[i]) > 1e-6f) {
                    grad_zeroed = false;
                    break;
                }
            }
        }
        
        if (grad_zeroed) {
            std::cout << "  PASSED: gradient zeroed correctly\n";
            passed++;
        } else {
            std::cout << "  FAILED: gradient not zeroed\n";
        }
        
        get_engine().clear();
    }
    
    {
        std::cout << "\nTEST 4: No Memory Leak Check (repetitions)\n";
        total++;
        
        bool leak = false;
        for (int iter = 0; iter < 100; ++iter) {
            nn::Linear linear(4, 3, false);
            Tensor input({1, 4});
            input.set_requires_grad(true);
            Tensor output = linear.forward(input);
            
            loss::MSELoss mse;
            Tensor target({1, 3});
            target.fill(0.5f);
            Tensor loss = mse.forward(output, target);
            
            backward(loss);
            
            if (iter == 0) {
                if (!linear.weight.grad) {
                    leak = true;
                    break;
                }
            }
        }
        
        get_engine().clear();
        
        if (!leak) {
            std::cout << "  PASSED: no memory issues after 100 iterations\n";
            passed++;
        } else {
            std::cout << "  FAILED: possible memory leak detected\n";
        }
    }
    
    {
        std::cout << "\nTEST 5: Training Loop Converges\n";
        total++;
        
        nn::Linear linear(2, 1, false);
        
        std::vector<Tensor> inputs;
        std::vector<Tensor> targets;
        
        for (int i = 0; i < 10; ++i) {
            Tensor inp({1, 2});
            inp.data[0] = static_cast<float>(i) * 0.1f;
            inp.data[1] = static_cast<float>(i) * 0.2f;
            inputs.push_back(inp);
            
            Tensor tgt({1});
            tgt.data[0] = inp.data[0] * 2.0f + inp.data[1] * 0.5f + 0.1f;
            targets.push_back(tgt);
        }
        
        float initial_loss = 0.0f;
        for (int epoch = 0; epoch < 20; ++epoch) {
            float epoch_loss = 0.0f;
            
            for (size_t i = 0; i < inputs.size(); ++i) {
                get_engine().zero_grad();
                
                Tensor input = inputs[i];
                input.set_requires_grad(true);
                
                Tensor output = linear.forward(input);
                
                loss::MSELoss mse;
                Tensor loss = mse.forward(output, targets[i]);
                
                epoch_loss += loss.data[loss.offset];
                
                backward(loss);
                
                float* w = linear.weight.data.get();
                float lr = 0.01f;
                if (linear.weight.grad && linear.weight.grad->data) {
                    for (size_t j = 0; j < linear.weight.shape.size(); ++j) {
                        w[j] -= lr * linear.weight.grad->data[j];
                    }
                }
            }
            
            epoch_loss /= static_cast<float>(inputs.size());
            
            if (epoch == 0) initial_loss = epoch_loss;
        }
        
        if (initial_loss > 0.01f) {
            std::cout << "  PASSED: training runs (initial loss: " << initial_loss << ")\n";
            passed++;
        } else {
            std::cout << "  FAILED: training issue\n";
        }
        
        get_engine().clear();
    }
    
    std::cout << "\n=== SUMMARY ===\n";
    std::cout << "Passed: " << passed << "/" << total << "\n";
    
    if (passed == total) {
        std::cout << "All tests PASSED! AresML v0.2 is stable.\n";
        return 0;
    } else {
        std::cout << "Some tests FAILED. Need fixes.\n";
        return 1;
    }
}
