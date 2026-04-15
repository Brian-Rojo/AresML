#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/Linear.hpp"
#include "loss/CrossEntropy.hpp"

using namespace aresml;

int main() {
    std::cout << "=== Test CrossEntropy Forward ===\n\n";
    
    nn::Linear linear(4, 10, false);
    
    Tensor input({1, 4});
    input.data[0] = 0.1f;
    input.data[1] = 0.2f;
    input.data[2] = 0.3f;
    input.data[3] = 0.4f;
    
    std::cout << "Forward: ";
    Tensor output = linear.forward(input);
    for (size_t i = 0; i < 5; ++i) {
        std::cout << output.data[i] << " ";
    }
    std::cout << "\n";
    
    Tensor target({1});
    target.data[0] = 3.0f;
    
    loss::CrossEntropyLoss loss_fn(true);
    Tensor loss = loss_fn.forward(output, target);
    
    std::cout << "Loss: " << loss.data[loss.offset] << "\n";
    
    if (std::isnan(loss.data[loss.offset]) || std::isinf(loss.data[loss.offset])) {
        std::cout << "FAILED: NaN/Inf in loss\n";
    } else {
        std::cout << "PASSED\n";
    }
    
    return 0;
}
