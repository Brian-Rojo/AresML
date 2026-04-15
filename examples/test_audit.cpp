#include <iostream>
#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/Linear.hpp"
#include "ops/Ops.hpp"
#include "loss/MSELoss.hpp"

using namespace aresml;

int main() {
    std::cout << "=== Test: Linear + Training Loop ===\n";
    
    zero_grad();
    
    nn::Linear linear(2, 1);
    
    Tensor x({1, 2});
    x.set_requires_grad(true);
    x.is_leaf = true;
    x.data[0] = 1.0f;
    x.data[1] = 2.0f;
    
    get_engine().register_leaf(&x);
    get_engine().register_leaf(&linear.weight);
    
    for (int step = 0; step < 3; ++step) {
        std::cout << "Step " << step << ":\n";
        
        Tensor y = linear.forward(x);
        std::cout << "  y = " << y.data[0] << "\n";
        
        loss::MSELoss loss_fn;
        Tensor target({1});
        target.data[0] = 1.0f;
        Tensor loss = loss_fn.forward(y, target);
        std::cout << "  loss = " << loss.data[0] << "\n";
        
        backward(loss);
        
        float grad = linear.weight.grad ? linear.weight.grad->data[0] : 0.0f;
        std::cout << "  grad = " << grad << "\n";
        
        zero_grad();
        
        grad = linear.weight.grad ? linear.weight.grad->data[0] : 0.0f;
        std::cout << "  after zero_grad = " << grad << "\n";
    }
    
    std::cout << "DONE\n";
    return 0;
}