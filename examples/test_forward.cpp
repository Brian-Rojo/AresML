#include <iostream>
#include <vector>
#include <random>
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
        
        Tensor embeddings({batch, seq_len, embed_dim});
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t idx = static_cast<size_t>(input_ids.data[input_ids.offset + b * seq_len + s]);
                for (size_t e = 0; e < embed_dim; ++e) {
                    embeddings.data[embeddings.offset + b * seq_len * embed_dim + s * embed_dim + e] = 
                        ((idx * 31 + e * 17) % 1000) / 1000.0f * 0.1f;
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
    std::cout << "=== Forward Pass Test v0.2 ===\n";
    
    size_t vocab_size = 16;
    size_t embed_dim = 32;
    size_t num_heads = 2;
    size_t num_layers = 1;
    size_t seq_len = 4;
    size_t batch_size = 2;
    
    std::cout << "Config: vocab=" << vocab_size << " embed=" << embed_dim 
              << " heads=" << num_heads << " layers=" << num_layers << "\n";
    
    try {
        std::cout << "Creating model...\n";
        SimpleModel model(vocab_size, embed_dim, num_heads, num_layers);
        
        std::cout << "Creating input...\n";
        Tensor input_ids({batch_size, seq_len});
        for (size_t i = 0; i < batch_size * seq_len; ++i) {
            input_ids.data[i] = static_cast<float>(i % vocab_size);
        }
        
        std::cout << "Running forward pass...\n";
        Tensor logits = model.forward(input_ids);
        
        std::cout << "Logits: [" << logits.shape.n << "D] ";
        for (size_t i = 0; i < logits.shape.n; ++i) {
            if (i) std::cout << "x";
            std::cout << logits.shape[i];
        }
        std::cout << " = " << logits.shape.size() << " elements\n";
        
        float max_logit = -1e9f;
        for (size_t i = 0; i < std::min(size_t(20), logits.shape.size()); ++i) {
            if (logits.data[logits.offset + i] > max_logit) max_logit = logits.data[logits.offset + i];
        }
        std::cout << "Max logit: " << max_logit << "\n";
        
        bool has_nan = false;
        for (size_t i = 0; i < logits.shape.size(); ++i) {
            float v = logits.data[logits.offset + i];
            if (std::isnan(v) || std::isinf(v)) {
                std::cout << "NaN/Inf at " << i << ": " << v << "\n";
                has_nan = true;
                break;
            }
        }
        
        if (!has_nan) {
            std::cout << "No NaN/Inf in logits - forward pass OK!\n";
        }
        
        std::cout << "\n=== Forward Test Complete ===\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
