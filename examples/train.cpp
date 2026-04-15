#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cassert>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/Linear.hpp"
#include "nn/TransformerBlock.hpp"
#include "nn/LMHead.hpp"
#include "optim/Adam.hpp"
#include "loss/CrossEntropy.hpp"

using namespace aresml;

struct SimpleModel {
    size_t vocab_size;
    size_t embed_dim;
    size_t num_heads;
    size_t num_layers;
    
    std::vector<nn::TransformerBlock*> blocks;
    nn::LMHead lm_head;
    
    std::vector<Tensor*> parameters;
    
    SimpleModel(size_t vocab_size, size_t embed_dim, size_t num_heads, size_t num_layers)
        : vocab_size(vocab_size), embed_dim(embed_dim), num_heads(num_heads), num_layers(num_layers),
          lm_head(vocab_size, embed_dim) {
        
        for (size_t i = 0; i < num_layers; ++i) {
            blocks.push_back(new nn::TransformerBlock(embed_dim, num_heads));
        }
        
        collect_params();
    }
    
    void collect_params() {
        parameters.clear();
        
        for (auto* block : blocks) {
            parameters.push_back(&block->attn_norm.weight);
            parameters.push_back(&block->ffn_norm.weight);
            parameters.push_back(&block->attention.q_proj.weight);
            parameters.push_back(&block->attention.k_proj.weight);
            parameters.push_back(&block->attention.v_proj.weight);
            parameters.push_back(&block->attention.out_proj.weight);
            parameters.push_back(&block->ffn.gate_proj.weight);
            parameters.push_back(&block->ffn.up_proj.weight);
            parameters.push_back(&block->ffn.down_proj.weight);
        }
        
        parameters.push_back(&lm_head.linear.weight);
    }
    
    Tensor forward(const Tensor& input_ids) {
        size_t batch = input_ids.shape[0];
        size_t seq_len = input_ids.shape[1];
        
        Tensor embeddings({batch, seq_len, embed_dim}, input_ids.requires_grad);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t idx = static_cast<size_t>(input_ids.data[input_ids.offset + b * seq_len + s]);
                for (size_t e = 0; e < embed_dim; ++e) {
                    embeddings.data[embeddings.offset + b * seq_len * embed_dim + s * embed_dim + e] = 
                        ((idx * 31 + e * 17) % 1000) / 1000.0f;
                }
            }
        }
        
        Tensor x = embeddings;
        
        for (auto* block : blocks) {
            x = block->forward(x);
        }
        
        Tensor logits = lm_head.forward(x);
        
        return logits;
    }
};

int main() {
    std::cout << "=== Training Test v0.2 ===\n";
    
    size_t vocab_size = 32;
    size_t embed_dim = 64;
    size_t num_heads = 4;
    size_t num_layers = 2;
    size_t seq_len = 8;
    size_t batch_size = 4;
    size_t num_epochs = 10;
    float lr = 0.001f;
    
    std::cout << "Config: vocab=" << vocab_size << " embed=" << embed_dim 
              << " heads=" << num_heads << " layers=" << num_layers << "\n";
    
    try {
        std::cout << "Creating model...\n";
        SimpleModel model(vocab_size, embed_dim, num_heads, num_layers);
        
        std::cout << "Setting up optimizer...\n";
        optim::Adam optimizer(model.parameters, lr);
        
        std::cout << "Creating training data...\n";
        std::vector<Tensor> inputs;
        std::vector<Tensor> targets;
        
        for (size_t i = 0; i < batch_size; ++i) {
            Tensor inp({seq_len});
            for (size_t j = 0; j < seq_len; ++j) {
                inp.data[j] = static_cast<float>(rand() % vocab_size);
            }
            inputs.push_back(inp);
            
            Tensor tgt({seq_len});
            for (size_t j = 0; j < seq_len; ++j) {
                tgt.data[j] = static_cast<float>((static_cast<size_t>(inp.data[j]) + 1) % vocab_size);
            }
            targets.push_back(tgt);
        }
        
        loss::CrossEntropyLoss loss_fn(true);
        
        std::cout << "Starting training loop...\n";
        
        for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
            float epoch_loss = 0.0f;
            
            for (size_t i = 0; i < batch_size; ++i) {
                optimizer.zero_grad();
                
                Tensor input = inputs[i];
                input.set_requires_grad(true);
                
                Tensor logits = model.forward(input);
                
                Tensor target = targets[i];
                
                Tensor loss = loss_fn.forward(logits, target);
                
                float loss_val = loss.data[loss.offset];
                epoch_loss += loss_val;
                
                backward(loss);
                
                optimizer.step();
            }
            
            epoch_loss /= static_cast<float>(batch_size);
            
            if (epoch % 2 == 0 || epoch == num_epochs - 1) {
                std::cout << "Epoch " << epoch << " | Loss: " << epoch_loss << "\n";
            }
            
            if (epoch_loss < 0.1f && epoch > 3) {
                std::cout << "Early stopping - loss converged!\n";
                break;
            }
        }
        
        std::cout << "\n=== Training Complete ===\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
