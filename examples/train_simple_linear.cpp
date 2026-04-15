#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/Linear.hpp"
#include "optim/Adam.hpp"
#include "loss/CrossEntropy.hpp"

using namespace aresml;

struct SimpleLinearModel {
    size_t in_features;
    size_t out_features;
    nn::Linear linear;
    std::vector<Tensor*> parameters;
    
    SimpleLinearModel(size_t in_features, size_t out_features)
        : in_features(in_features), out_features(out_features),
          linear(in_features, out_features, false) {
        parameters.push_back(&linear.weight);
    }
    
    Tensor forward(const Tensor& x) {
        return linear.forward(x);
    }
};

int main() {
    std::cout << "=== Simple Training Test ===\n";
    
    size_t in_features = 8;
    size_t out_features = 4;
    size_t batch_size = 2;
    size_t num_epochs = 20;
    float lr = 0.01f;
    
    try {
        std::cout << "Creating model...\n";
        SimpleLinearModel model(in_features, out_features);
        
        std::cout << "Creating data...\n";
        std::vector<Tensor> inputs;
        std::vector<Tensor> targets;
        
        for (size_t i = 0; i < batch_size; ++i) {
            Tensor inp({1, in_features});
            for (size_t j = 0; j < in_features; ++j) {
                inp.data[inp.offset + j] = (rand() % 100) / 100.0f;
            }
            inputs.push_back(inp);
            
            Tensor tgt({1});
            tgt.data[tgt.offset + 0] = static_cast<float>(rand() % out_features);
            targets.push_back(tgt);
        }
        
        std::cout << "Setting up optimizer...\n";
        optim::Adam optimizer(model.parameters, lr);
        
        loss::CrossEntropyLoss loss_fn(true);
        
        std::cout << "Training loop...\n";
        
        for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
            float epoch_loss = 0.0f;
            
            for (size_t i = 0; i < batch_size; ++i) {
                optimizer.zero_grad();
                
                Tensor input = inputs[i];
                input.set_requires_grad(true);
                
                Tensor logits = model.forward(input);
                
                Tensor target = targets[i];
                
                Tensor loss = loss_fn.forward(logits, target);
                
                float loss_val = loss.data[loss.offset];
                epoch_loss += loss_val;
                
                backward(loss);
                
                optimizer.step();
            }
            
            epoch_loss /= static_cast<float>(batch_size);
            std::cout << "Epoch " << epoch << " | Loss: " << epoch_loss << "\n";
            
            if (epoch_loss < 0.1f && epoch > 3) break;
        }
        
        std::cout << "\n=== Training Complete ===\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
