#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/Linear.hpp"

using namespace aresml;

struct MSELoss {
    Tensor forward(const Tensor& pred, const Tensor& target) {
        size_t n = pred.shape.size();
        
        Tensor loss({1});
        loss.requires_grad = true;
        
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            float diff = pred.data[pred.offset + i] - target.data[target.offset + i];
            sum += diff * diff;
        }
        loss.data[0] = sum / static_cast<float>(n);
        
        if (pred.requires_grad) {
            loss.inputs = {const_cast<Tensor*>(&pred), const_cast<Tensor*>(&target)};
            
            struct MSELossOp : Operation {
                Tensor* pred;
                Tensor* target;
                size_t n;
                MSELossOp(Tensor* p, Tensor* t, size_t n_) : pred(p), target(t), n(n_) {}
                std::vector<Tensor*> get_inputs() const override {
                    return {pred, target};
                }
                void backward(Tensor& grad) override {
                    if (!pred->grad || !pred->grad->data) {
                        pred->grad = std::make_shared<Tensor>(pred->shape);
                        pred->grad->zero_();
                    }
                    float* g = pred->grad->data.get() + pred->grad->offset;
                    float scale = 2.0f / static_cast<float>(n);
                    for (size_t i = 0; i < n; ++i) {
                        float diff = pred->data[pred->offset + i] - target->data[target->offset + i];
                        g[i] += scale * diff;
                    }
                }
                std::unique_ptr<Operation> clone() const override {
                    return std::make_unique<MSELossOp>(pred, target, n);
                }
            };
            
            loss.op = std::make_unique<MSELossOp>(const_cast<Tensor*>(&pred), const_cast<Tensor*>(&target), n);
            get_engine().register_operation(&loss, {const_cast<Tensor*>(&pred)});
        }
        
        return loss;
    }
};

int main() {
    std::cout << "=== MSE Loss Training Test ===\n\n";
    
    nn::Linear linear(4, 3, false);
    std::cout << "linear.weight: " << &linear.weight << "\n\n";
    
    Tensor input({1, 4});
    input.data[0] = 0.1f;
    input.data[1] = 0.2f;
    input.data[2] = 0.3f;
    input.data[3] = 0.4f;
    input.set_requires_grad(true);
    
    std::cout << "=== Forward pass ===\n";
    Tensor output = linear.forward(input);
    
    std::cout << "output.op: " << (output.op ? "exists" : "nullptr") << "\n";
    std::cout << "output.inputs[0]: " << output.inputs[0] << "\n\n";
    
    Tensor target({1, 3});
    target.data[0] = 0.5f;
    target.data[1] = 0.5f;
    target.data[2] = 0.5f;
    
    MSELoss mse;
    Tensor loss = mse.forward(output, target);
    
    std::cout << "loss value: " << loss.data[0] << "\n";
    std::cout << "loss.op: " << (loss.op ? "exists" : "nullptr") << "\n\n";
    
    std::cout << "=== Backward pass ===\n";
    backward(loss);
    
    std::cout << "\n=== Check weight.grad ===\n";
    if (linear.weight.grad && linear.weight.grad->data) {
        std::cout << "SUCCESS! weight has gradient\n";
        std::cout << "weight.grad: ";
        for (size_t i = 0; i < linear.weight.shape.size(); ++i) {
            std::cout << linear.weight.grad->data[i] << " ";
        }
        std::cout << "\n";
    } else {
        std::cout << "FAIL: weight.grad is " << linear.weight.grad << "\n";
    }
    
    return 0;
}
