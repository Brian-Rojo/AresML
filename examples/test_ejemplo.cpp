#include <iostream>
#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/Linear.hpp"
#include "loss/MSELoss.hpp"

using namespace aresml;

int main() {
    std::cout << "=== Ejemplo AresML ===\n";
    
    // Crear modelo
    nn::Linear linear(4, 2, false);
    
    // Input
    Tensor x({2, 4});
    x.set_requires_grad(true);
    x.is_leaf = true;
    
    for (int i = 0; i < 8; ++i) x.data[i] = (float)i * 0.1f;
    
    // Forward
    Tensor y = linear.forward(x);
    
    // Target
    Tensor target({2, 2});
    target.zero_();
    
    // Loss
    loss::MSELoss mse;
    Tensor loss = mse.forward(y, target);
    
    std::cout << "Loss: " << loss.data[0] << "\n";
    
    // Backward
    backward(loss);
    
    std::cout << "x.grad: " << (x.grad ? "OK" : "NULL") << "\n";
    std::cout << "weight.grad: " << (linear.weight.grad ? "OK" : "NULL") << "\n";
    
    return 0;
}