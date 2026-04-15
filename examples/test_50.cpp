#include <iostream>
#include <vector>
#include <cmath>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/Linear.hpp"
#include "loss/MSELoss.hpp"

using namespace aresml;

int main() {
    std::cout << "=== Test 50 steps ===\n\n";
    
    nn::Linear linear(2, 1, false);
    
    for (int step = 0; step < 50; ++step) {
        Tensor input({1, 2});
        input.data[0] = 0.5f;
        input.data[1] = 0.3f;
        input.set_requires_grad(true);
        
        Tensor target({1});
        target.data[0] = 0.8f;
        
        Tensor output = linear.forward(input);
        loss::MSELoss mse;
        Tensor loss = mse.forward(output, target);
        
        backward(loss);
        
        float lr = 0.01f;
        if (linear.weight.grad && linear.weight.grad->data) {
            linear.weight.data[0] -= lr * linear.weight.grad->data[0];
            linear.weight.data[1] -= lr * linear.weight.grad->data[1];
        }
        
        get_engine().clear();
        
        if (step % 10 == 0) {
            std::cout << "Step " << step << ": loss = " << loss.data[loss.offset] << "\n";
        }
    }
    
    std::cout << "Done!\n";
    return 0;
}
