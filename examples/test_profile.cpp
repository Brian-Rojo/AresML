#include <iostream>
#include <iomanip>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/Linear.hpp"
#include "loss/MSELoss.hpp"
#include "optim/SGD.hpp"
#include "utils/Profiler.hpp"

using namespace aresml;

int main() {
    std::cout << "\n=== AresML Performance Profile ===\n\n";
    
    // Enable profiler
    PROFILE_ENABLE();
    
    // Simple training setup
    size_t batch_size = 32;
    size_t input_size = 128;
    size_t hidden_size = 64;
    size_t output_size = 10;
    size_t num_iters = 100;
    
    // Create layers
    nn::Linear layer1(input_size, hidden_size);
    nn::Linear layer2(hidden_size, output_size);
    loss::MSELoss loss_fn;
    optim::SGD optimizer({&layer1.weight, &layer1.bias, &layer2.weight, &layer2.bias}, 0.01f);
    
    // Create dummy data
    Tensor input({batch_size, input_size});
    for (size_t i = 0; i < input.shape.size(); ++i) {
        input.data[i] = (float)(i % 10) / 10.0f;
    }
    input.set_requires_grad(true);
    
    Tensor target({batch_size, output_size});
    for (size_t i = 0; i < target.shape.size(); ++i) {
        target.data[i] = (float)(i % 5) / 5.0f;
    }
    
    std::cout << "Configuration:\n";
    std::cout << "  Batch size: " << batch_size << "\n";
    std::cout << "  Input size: " << input_size << "\n";
    std::cout << "  Hidden size: " << hidden_size << "\n";
    std::cout << "  Output size: " << output_size << "\n";
    std::cout << "  Iterations: " << num_iters << "\n\n";
    
    std::cout << "Running training loop with profiling...\n\n";
    
    float final_loss = 0.0f;
    
    for (size_t iter = 0; iter < num_iters; ++iter) {
        zero_grad();
        
        // Forward
        Tensor h = layer1.forward(input);
        Tensor out = layer2.forward(h);
        
        // Loss
        Tensor loss = loss_fn.forward(out, target);
        final_loss = loss.data[0];
        
        // Backward
        backward(loss);
        
        // Optimizer step
        optimizer.step();
        
        if ((iter + 1) % 20 == 0) {
            std::cout << "Iter " << iter + 1 << ": loss = " << std::fixed << std::setprecision(6) << final_loss << "\n";
        }
    }
    
    std::cout << "\n";
    PROFILE_REPORT();
    
    std::cout << "\nFinal loss: " << std::fixed << std::setprecision(6) << final_loss << "\n";
    std::cout << "Test completed successfully!\n";
    
    return 0;
}
