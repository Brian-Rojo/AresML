#include "../engine/gpt/GPTEngine.hpp"
#include "../engine/gpt/GPTDataset.hpp"
#include "../core/Autograd.hpp"
#include <iostream>
#include <cassert>
#include <iomanip>
#include <sstream>

using namespace aresml;
using namespace aresml::engine::gpt;

int main() {
    std::cout << "=== GPT Generation Test ===" << std::endl;
    
    size_t vocab_size = 64;
    size_t embed_dim = 64;
    size_t num_heads = 2;
    size_t num_layers = 1;
    size_t max_seq_len = 32;
    
    GPTEngine engine(vocab_size, embed_dim, num_heads, num_layers, max_seq_len);
    
    GPTDataset dataset = GPTDataset::simple_dataset(20, 8, vocab_size);
    
    std::cout << "Training for generation..." << std::endl;
    
    float initial_loss = 0.0f;
    size_t train_steps = 30;
    
    for (size_t step = 0; step < train_steps; ++step) {
        float step_loss = 0.0f;
        
        for (size_t i = 0; i < dataset.size(); ++i) {
            auto [input, target] = dataset.get(i);
            
            get_engine().zero_grad();
            
            Tensor logits = engine.model->forward(input);
            
            Tensor logits_flat = logits.view({logits.shape[0] * logits.shape[1], logits.shape[2]});
            Tensor target_flat = target.view({target.shape[0] * target.shape[1]});
            
            loss::CrossEntropyLoss loss_fn;
            Tensor loss_tensor = loss_fn.forward(logits_flat, target_flat);
            
            float batch_loss = 0.0f;
            for (size_t j = 0; j < loss_tensor.shape.size(); ++j) {
                batch_loss += loss_tensor.data[loss_tensor.offset + j];
            }
            step_loss += batch_loss;
            
            get_engine().backward(&loss_tensor);
            
            engine.trainer->optimizer->step();
        }
        
        step_loss /= static_cast<float>(dataset.size());
        
        if (step == 0) initial_loss = step_loss;
        
        if (step % 10 == 0) {
            std::cout << "Step " << step << " | Loss: " << std::fixed << std::setprecision(4) << step_loss << std::endl;
        }
    }
    
    std::cout << "\n=== Generating Text ===" << std::endl;
    
    std::vector<int> prompt = {1, 2, 3};
    int max_tokens = 10;
    
    std::cout << "Prompt: ";
    for (size_t i = 0; i < prompt.size(); ++i) {
        std::cout << prompt[i] << " ";
    }
    std::cout << std::endl;
    
    std::vector<int> generated = engine.generate_tokens(prompt, max_tokens);
    
    std::cout << "Generated: ";
    for (size_t i = 0; i < generated.size(); ++i) {
        std::cout << generated[i] << " ";
    }
    std::cout << std::endl;
    
    if (generated.size() > prompt.size()) {
        std::cout << "Generated " << (generated.size() - prompt.size()) << " new tokens" << std::endl;
        std::cout << "=== GPT Generation Test PASSED ===" << std::endl;
        return 0;
    } else {
        std::cout << "=== GPT Generation Test FAILED ===" << std::endl;
        return 1;
    }
}
