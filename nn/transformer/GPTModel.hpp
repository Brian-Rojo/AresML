#pragma once

#include "../../core/Tensor.hpp"
#include "TokenEmbedding.hpp"
#include "PositionalEncoding.hpp"
#include "GPTBlock.hpp"
#include "../Linear.hpp"

namespace aresml {
namespace nn {
namespace transformer {

struct GPTModel {
    size_t vocab_size;
    size_t embed_dim;
    size_t num_heads;
    size_t num_layers;
    size_t max_seq_len;
    size_t hidden_dim;
    
    TokenEmbedding token_embedding;
    PositionalEncodingLearned pos_encoding;
    std::vector<GPTBlock> blocks;
    LayerNorm final_norm;
    Linear* lm_head;
    
    GPTModel(size_t vocab_size, size_t embed_dim, size_t num_heads, size_t num_layers, size_t max_seq_len = 64)
        : vocab_size(vocab_size), embed_dim(embed_dim), num_heads(num_heads),
          num_layers(num_layers), max_seq_len(max_seq_len), hidden_dim(embed_dim * 4),
          token_embedding(vocab_size, embed_dim),
          pos_encoding(max_seq_len, embed_dim),
          final_norm(embed_dim),
          lm_head(new Linear("lm_head", embed_dim, vocab_size, false)) {

        for (size_t i = 0; i < num_layers; ++i) {
            blocks.emplace_back(embed_dim, num_heads, hidden_dim);
        }

        // lm_head parameters are already registered by Linear constructor
    }
    
    ~GPTModel() {
        delete lm_head;
    }
    
    Tensor forward(const Tensor& input_ids) {
        size_t batch_size = 1;
        size_t seq_len = 1;
        
        if (input_ids.shape.n >= 1) batch_size = input_ids.shape[0];
        if (input_ids.shape.n >= 2) seq_len = input_ids.shape[1];
        
        Tensor embeddings = token_embedding.forward(input_ids);
        
        Tensor encoded = pos_encoding.forward(embeddings);
        
        Tensor hidden = encoded;
        
        for (auto& block : blocks) {
            hidden = block.forward(hidden);
        }
        
        hidden = final_norm.forward(hidden);
        
        size_t B = 1, S = 1;
        if (hidden.shape.n >= 1) B = hidden.shape[0];
        if (hidden.shape.n >= 2) S = hidden.shape[1];
        
        Tensor hidden_2d = hidden.view({B * S, embed_dim});
        
        Tensor logits_2d = lm_head->forward(hidden_2d);
        
        Tensor logits = logits_2d.view({B, S, vocab_size});
        
        return logits;
    }
    
    Tensor generate(const std::vector<int>& prompt, int max_new_tokens, float temperature = 1.0f) {
        std::vector<int> generated = prompt;
        
        for (size_t i = 0; i < static_cast<size_t>(max_new_tokens); ++i) {
            if (generated.size() >= max_seq_len) break;
            
            Tensor input_tensor = Tensor(Shape{1, generated.size()}, false);
            for (size_t j = 0; j < generated.size(); ++j) {
                input_tensor.data[input_tensor.offset + j] = static_cast<float>(generated[j]);
            }
            
            Tensor logits = forward(input_tensor);
            
            size_t last_idx = logits.shape[1] - 1;
            Tensor last_logits = logits.view({logits.shape[0] * logits.shape[1], logits.shape[2]});
            last_logits = last_logits.view({1, logits.shape[2]});
            
            float max_val = -1e9f;
            for (size_t j = 0; j < vocab_size; ++j) {
                max_val = std::max(max_val, last_logits.data[last_logits.offset + j]);
            }
            
            float sum = 0.0f;
            std::vector<float> probs(vocab_size);
            for (size_t j = 0; j < vocab_size; ++j) {
                float val = last_logits.data[last_logits.offset + j];
                if (temperature > 0.0f) {
                    val = (val - max_val) / temperature;
                }
                probs[j] = std::exp(val);
                sum += probs[j];
            }
            
            float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            float cumulative = 0.0f;
            int next_token = 0;
            for (size_t j = 0; j < vocab_size; ++j) {
                probs[j] /= sum;
                cumulative += probs[j];
                if (r <= cumulative) {
                    next_token = static_cast<int>(j);
                    break;
                }
            }
            
            generated.push_back(next_token);
        }
        
        Tensor output(Shape{1, generated.size()}, false);
        for (size_t i = 0; i < generated.size(); ++i) {
            output.data[i] = static_cast<float>(generated[i]);
        }
        
        return output;
    }
};

}
}
}