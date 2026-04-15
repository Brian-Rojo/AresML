#include <iostream>
#include <fstream>
#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "core/Checkpoint.hpp"
#include "nn/Linear.hpp"
#include "loss/MSELoss.hpp"

using namespace aresml;

int main() {
    std::cout << "=== Checkpoint Save/Load Test ===\n\n";
    
    nn::Linear linear("model.linear", 4, 3, true);
    
    for (size_t i = 0; i < linear.weight.shape.size(); ++i) {
        linear.weight.data[i] = static_cast<float>(i) * 0.1f;
    }
    
    std::cout << "Original weight (first 4): ";
    for (size_t i = 0; i < 4; ++i) {
        std::cout << linear.weight.data[i] << " ";
    }
    std::cout << "\n";
    
    auto state = linear.state_dict();
    std::cout << "State dict keys: " << state.size() << " tensors\n";
    for (auto& p : state) {
        std::cout << "  - " << p.first << "\n";
    }
    
    std::string path = "/tmp/test_model.ares";
    
    std::cout << "\nSaving to " << path << "...\n";
    Checkpoint::save_model(path, linear);
    
    Checkpoint::print_info(path);
    
    nn::Linear linear2("model.linear", 4, 3, true);
    for (size_t i = 0; i < linear2.weight.shape.size(); ++i) {
        linear2.weight.data[i] = 999.0f;
    }
    
    std::cout << "\nBefore load weight[0]: " << linear2.weight.data[0] << "\n";
    
    std::cout << "Loading from " << path << "...\n";
    Checkpoint::load_model(path, linear2);
    
    std::cout << "After load weight[0]: " << linear2.weight.data[0] << "\n";
    
    bool match = true;
    for (size_t i = 0; i < linear.weight.shape.size(); ++i) {
        if (std::abs(linear.weight.data[i] - linear2.weight.data[i]) > 1e-5f) {
            match = false;
            break;
        }
    }
    
    if (match) {
        std::cout << "\nPASSED: Model saved and loaded correctly\n";
    } else {
        std::cout << "\nFAILED: Weights don't match\n";
    }
    
    std::remove(path.c_str());
    
    std::cout << "\n=== ALL TESTS COMPLETED ===\n";
    return 0;
}