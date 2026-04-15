#include <iostream>
#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/Attention.hpp"
#include "loss/MSELoss.hpp"

using namespace aresml;

int main() {
    std::cout << "=== Attention Backward Test ===\n";
    
    nn::Attention attn(8, 2, false);
    
    Tensor x({2, 4, 8});
    x.set_requires_grad(true);
    x.is_leaf = true;
    
    for (int i = 0; i < 64; ++i) {
        x.data[i] = (float)(i % 8) * 0.1f;
    }
    
    get_engine().register_leaf(&x);
    get_engine().register_leaf(&attn.q_proj.weight);
    get_engine().register_leaf(&attn.k_proj.weight);
    get_engine().register_leaf(&attn.v_proj.weight);
    get_engine().register_leaf(&attn.out_proj.weight);
    
    std::cout << "Forward pass...\n";
    Tensor out = attn.forward(x);
    
    loss::MSELoss loss_fn;
    Tensor target({out.shape[0], out.shape[1], out.shape[2]});
    target.zero_();
    
    std::cout << "Computing loss...\n";
    Tensor loss = loss_fn.forward(out, target);
    std::cout << "Loss: " << loss.data[0] << "\n";
    
    std::cout << "Backward pass...\n";
    backward(loss);
    
    // Input gradient
    if (x.grad) {
        float max_grad = 0.0f;
        for (int i = 0; i < 64; ++i) {
            max_grad = std::max(max_grad, std::abs(x.grad->data[i]));
        }
        std::cout << "Input grad max: " << max_grad << "\n";
        if (max_grad > 1e-6f) {
            std::cout << "PASS: Gradients flow to input\n";
        } else {
            std::cout << "FAIL: Zero gradients on input\n";
        }
    }
    
    // Q weight gradient
    if (attn.q_proj.weight.grad) {
        float w_grad = 0.0f;
        for (int i = 0; i < 64; ++i) {
            w_grad += std::abs(attn.q_proj.weight.grad->data[i]);
        }
        std::cout << "Q weight grad sum: " << w_grad << "\n";
    }
    
    // K weight gradient
    if (attn.k_proj.weight.grad) {
        float w_grad = 0.0f;
        for (int i = 0; i < 64; ++i) {
            w_grad += std::abs(attn.k_proj.weight.grad->data[i]);
        }
        std::cout << "K weight grad sum: " << w_grad << "\n";
    }
    
    // V weight gradient
    if (attn.v_proj.weight.grad) {
        float w_grad = 0.0f;
        for (int i = 0; i < 64; ++i) {
            w_grad += std::abs(attn.v_proj.weight.grad->data[i]);
        }
        std::cout << "V weight grad sum: " << w_grad << "\n";
    }
    
    std::cout << "\n=== ALL TESTS COMPLETED ===\n";
    return 0;
}