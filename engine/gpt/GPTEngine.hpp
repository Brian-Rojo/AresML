#pragma once

#include "../../nn/transformer/GPTModel.hpp"
#include "GPTTrainer.hpp"
#include "GPTInference.hpp"
#include "GPTDataset.hpp"
#include "Tokenizer.hpp"
#include <iostream>
#include <memory>

namespace aresml {
namespace engine {
namespace gpt {

struct GPTEngine {
    std::unique_ptr<aresml::nn::transformer::GPTModel> model;
    std::unique_ptr<GPTTrainer> trainer;
    std::unique_ptr<GPTInference> inference;
    Tokenizer tokenizer;
    
    size_t vocab_size;
    size_t embed_dim;
    size_t num_heads;
    size_t num_layers;
    size_t max_seq_len;
    
    GPTEngine(size_t vocab_size = 256, size_t embed_dim = 128, size_t num_heads = 4, 
              size_t num_layers = 2, size_t max_seq_len = 64)
        : vocab_size(vocab_size), embed_dim(embed_dim), num_heads(num_heads),
          num_layers(num_layers), max_seq_len(max_seq_len) {
        
        model = std::make_unique<aresml::nn::transformer::GPTModel>(
            vocab_size, embed_dim, num_heads, num_layers, max_seq_len);
        
        trainer = std::make_unique<GPTTrainer>(model.get(), 1e-3f);
        
        inference = std::make_unique<GPTInference>(model.get(), tokenizer, 1.0f, 10, 0.9f);
    }
    
    void train_step(const Tensor& input, const Tensor& target) {
        if (trainer) {
            trainer->train_step(input, target);
        }
    }
    
    void train_epoch(const GPTDataset& dataset, size_t batch_size = 1) {
        if (trainer) {
            trainer->train_epoch(dataset, batch_size);
        }
    }
    
    Tensor generate(const std::vector<int>& prompt, int max_tokens) {
        if (!model) return Tensor();
        
        Tensor input_tensor(Shape{1, prompt.size()}, false);
        for (size_t i = 0; i < prompt.size(); ++i) {
            input_tensor.data[i] = static_cast<float>(prompt[i]);
        }
        
        std::vector<int> generated = prompt;
        
        for (int i = 0; i < max_tokens; ++i) {
            if (generated.size() >= max_seq_len) break;
            
            Tensor inp(Shape{1, generated.size()}, false);
            for (size_t j = 0; j < generated.size(); ++j) {
                inp.data[j] = static_cast<float>(generated[j]);
            }
            
            Tensor logits = model->forward(inp);
            
            Tensor last = logits.view({logits.shape[0] * logits.shape[1], logits.shape[2]});
            last = last.view({1, logits.shape[2]});
            
            int next = aresml::engine::gpt::Sampling::sample_temperature(last, 1.0f);
            generated.push_back(next);
            
            if (next == 0) break;
        }
        
        Tensor output(Shape{1, generated.size()}, false);
        for (size_t i = 0; i < generated.size(); ++i) {
            output.data[i] = static_cast<float>(generated[i]);
        }
        
        return output;
    }
    
    std::vector<int> generate_tokens(const std::vector<int>& prompt, int max_tokens) {
        if (!model) return {};
        
        std::vector<int> generated = prompt;
        
        for (int i = 0; i < max_tokens; ++i) {
            if (generated.size() >= max_seq_len) break;
            
            Tensor inp(Shape{1, generated.size()}, false);
            for (size_t j = 0; j < generated.size(); ++j) {
                inp.data[j] = static_cast<float>(generated[j]);
            }
            
            Tensor logits = model->forward(inp);
            
            Tensor last = logits.view({logits.shape[0] * logits.shape[1], logits.shape[2]});
            last = last.view({1, logits.shape[2]});
            
            int next = aresml::engine::gpt::Sampling::sample_temperature(last, 1.0f);
            generated.push_back(next);
            
            if (next == 0) break;
        }
        
        return generated;
    }
    
    void print_info() {
        std::cout << "=== GPT Engine Info ===" << std::endl;
        std::cout << "Vocab Size: " << vocab_size << std::endl;
        std::cout << "Embed Dim: " << embed_dim << std::endl;
        std::cout << "Num Heads: " << num_heads << std::endl;
        std::cout << "Num Layers: " << num_layers << std::endl;
        std::cout << "Max Seq Len: " << max_seq_len << std::endl;
    }
};

}
}
}
