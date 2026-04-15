#include "../nn/transformer/GPTModel.hpp"
#include "../core/Autograd.hpp"
#include "../loss/CrossEntropy.hpp"
#include <iostream>

using namespace aresml;
using namespace aresml::nn::transformer;

int main() {
    GPTModel model(16, 16, 2, 1, 8);
    
    Tensor input(Shape{1, 4});
    input.set_requires_grad(true);
    input.is_leaf = true;
    get_engine().register_leaf(&input);
    
    for (int i = 0; i < 4; ++i) input.data[i] = (float)(i % 4);
    
    Tensor target(Shape{1, 4}, false);
    for (int i = 0; i < 4; ++i) target.data[i] = (float)((i + 1) % 4);
    
    get_engine().zero_grad();
    
    Tensor logits = model.forward(input);
    
    if (input.requires_grad && logits.requires_grad) {
        logits.inputs.push_back(&input);
    }
    
    Tensor logits_flat = logits.view({4, 16});
    
    loss::CrossEntropyLoss loss_fn;
    Tensor loss = loss_fn.forward(logits_flat, target);
    
    get_engine().backward(&loss);
    
    std::cout << "=== Gradient VALUES ===" << std::endl;
    
    std::cout << "input.grad: ";
    if (input.grad && input.grad->data) {
        float max = 0;
        for (int i = 0; i < 4; ++i) {
            max = std::max(max, std::abs(input.grad->data[i]));
        }
        std::cout << "max=" << max;
    } else {
        std::cout << "null";
    }
    std::cout << std::endl;
    
    std::cout << "token_embedding.weight.grad: ";
    if (model.token_embedding.weight.grad && model.token_embedding.weight.grad->data) {
        float* g = model.token_embedding.weight.grad->data.get();
        float max = 0;
        for (int i = 0; i < 16; ++i) max = std::max(max, std::abs(g[i]));
        std::cout << "max=" << max;
    } else {
        std::cout << "null";
    }
    std::cout << std::endl;
    
    std::cout << "pos_encoding.weight.grad: ";
    if (model.pos_encoding.weight.grad && model.pos_encoding.weight.grad->data) {
        float* g = model.pos_encoding.weight.grad->data.get();
        float max = 0;
        for (int i = 0; i < 16; ++i) max = std::max(max, std::abs(g[i]));
        std::cout << "max=" << max;
    } else {
        std::cout << "null";
    }
    std::cout << std::endl;
    
    std::cout << "Done!" << std::endl;
    return 0;
}
