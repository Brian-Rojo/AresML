#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cassert>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/Linear.hpp"
#include "optim/Adam.hpp"
#include "loss/CrossEntropy.hpp"

using namespace aresml;

int main() {
    std::cout << "=== Debug Test ===\n";
    set_debug(true);
    get_engine().set_debug(true);
    
    try {
        std::cout << "1. Creating model...\n";
        size_t in_features = 8;
        size_t out_features = 16;
        
        nn::Linear model(in_features, out_features);
        
        std::cout << "2. Creating input tensor...\n";
        Tensor input({2, in_features});
        input.set_requires_grad(true);
        for (size_t i = 0; i < input.shape.size(); ++i) {
            input.data[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        std::cout << "   Input shape: " << input.shape.n << "D, size=" << input.shape.size() << "\n";
        std::cout << "   Input requires_grad: " << input.requires_grad << "\n";
        
        std::cout << "3. Calling model.forward()...\n";
        Tensor logits = model.forward(input);
        std::cout << "   Logits shape: " << logits.shape.n << "D, size=" << logits.shape.size() << "\n";
        
        std::cout << "4. Creating targets...\n";
        Tensor targets({2});
        for (size_t i = 0; i < 2; ++i) {
            targets.data[i] = static_cast<float>(i % out_features);
        }
        
        std::cout << "5. Calling loss.forward()...\n";
        loss::CrossEntropyLoss loss_fn(true);
        Tensor loss = loss_fn.forward(logits, targets);
        std::cout << "   Loss shape: " << loss.shape.n << "D, size=" << loss.shape.size() << "\n";
        
        std::cout << "6. Calling backward()...\n";
        backward(loss);
        
        std::cout << "=== Gradient Check ===\n";
        if (model.weight.grad && model.weight.grad->data) {
            float max_g = 0, mean_g = 0;
            for (size_t i = 0; i < model.weight.grad->shape.size(); ++i) {
                float g = std::abs(model.weight.grad->data[i]);
                max_g = std::max(max_g, g);
                mean_g += g;
            }
            mean_g /= model.weight.grad->shape.size();
            std::cout << "weight.grad: max=" << max_g << " mean=" << mean_g << "\n";
        } else {
            std::cout << "weight.grad: NULL\n";
        }
        
        std::cout << "7. Done!\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
