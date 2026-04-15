#include "../engine/gpt/GPTEngine.hpp"
#include "../engine/gpt/GPTDataset.hpp"
#include "../core/Autograd.hpp"
#include "../ops/Ops.hpp"
#include <iostream>
#include <cassert>
#include <iomanip>

using namespace aresml;
using namespace aresml::engine::gpt;

int main() {
    std::cout << "=== GPT Overfit Test (Simplified) ===" << std::endl;
    
    size_t vocab_size = 16;
    size_t embed_dim = 16;
    size_t num_heads = 2;
    size_t num_layers = 1;
    size_t max_seq_len = 8;
    
    GPTEngine engine(vocab_size, embed_dim, num_heads, num_layers, max_seq_len);
    
    GPTDataset dataset = GPTDataset::simple_dataset(4, 4, vocab_size);
    
    std::cout << "Dataset size: " << dataset.size() << std::endl;
    
    float initial_loss = 0.0f;
    float final_loss = 0.0f;
    size_t num_epochs = 30;
    
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        
        for (size_t i = 0; i < dataset.size(); ++i) {
            auto [input, target] = dataset.get(i);
            
            get_engine().zero_grad();
            
            Tensor logits = engine.model->forward(input);
            
            size_t B = logits.shape[0];
            size_t S = 1;
            if (logits.shape.n >= 2) S = logits.shape[1];
            
            Tensor logits_flat = logits.clone().view({B * S, logits.shape[logits.shape.n - 1]});
            logits_flat.set_requires_grad(true);
            input.set_requires_grad(true);
            Tensor target_flat = target.view({target.shape[0] * target.shape[1]});
            
            loss::CrossEntropyLoss loss_fn;
            Tensor loss_tensor = loss_fn.forward(logits_flat, target_flat);
            
            std::cout << "Forward chain:" << std::endl;
            std::cout << "  logits_flat op: " << (logits_flat.op ? "yes" : "no") << std::endl;
            std::cout << "  logits_flat.requires_grad: " << logits_flat.requires_grad << std::endl;
            
            float batch_loss = 0.0f;
            for (size_t j = 0; j < loss_tensor.shape.size(); ++j) {
                batch_loss += loss_tensor.data[loss_tensor.offset + j];
            }
            epoch_loss += batch_loss;
            
            get_engine().backward(&loss_tensor);
            
            std::cout << "Backward pass complete." << std::endl;
            
            if (engine.model->token_embedding.weight.grad && engine.model->token_embedding.weight.grad->data) {
                std::cout << "  Has grad" << std::endl;
            }
            
            if (engine.model->token_embedding.weight.inputs.size() > 0) {
                std::cout << "  Token embedding has inputs" << std::endl;
            }
            
            engine.trainer->optimizer->step();
            engine.trainer->optimizer->zero_grad();
        }
        
        epoch_loss /= static_cast<float>(dataset.size());
        
        if (epoch == 0) initial_loss = epoch_loss;
        if (epoch == num_epochs - 1) final_loss = epoch_loss;
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << std::fixed << std::setprecision(4) << epoch_loss << std::endl;
        }
    }
    
    std::cout << std::endl;
    std::cout << "Initial Loss: " << std::fixed << std::setprecision(4) << initial_loss << std::endl;
    std::cout << "Final Loss: " << std::fixed << std::setprecision(4) << final_loss << std::endl;
    
    if (final_loss < initial_loss) {
        std::cout << "=== GPT Overfit Test PASSED ===" << std::endl;
        return 0;
    } else {
        std::cout << "=== GPT Overfit Test FAILED ===" << std::endl;
        return 1;
    }
}
