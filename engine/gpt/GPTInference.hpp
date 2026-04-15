#pragma once

#include "../../nn/transformer/GPTModel.hpp"
#include "Sampling.hpp"
#include "Tokenizer.hpp"
#include <vector>
#include <iostream>

namespace aresml {
namespace engine {
namespace gpt {

struct GPTInference {
    aresml::nn::transformer::GPTModel* model;
    Tokenizer tokenizer;
    float temperature;
    int top_k;
    float top_p;
    
    GPTInference(aresml::nn::transformer::GPTModel* m, const Tokenizer& tok, float temp = 1.0f, int tk = 10, float tp = 0.9f)
        : model(m), tokenizer(tok), temperature(temp), top_k(tk), top_p(tp) {}
    
    std::vector<int> generate(const std::vector<int>& prompt, int max_new_tokens) {
        if (!model) return {};
        
        std::vector<int> generated = prompt;
        
        for (int i = 0; i < max_new_tokens; ++i) {
            if (generated.size() >= model->max_seq_len) break;
            
            Tensor input_tensor(Shape{1, generated.size()}, false);
            for (size_t j = 0; j < generated.size(); ++j) {
                input_tensor.data[j] = static_cast<float>(generated[j]);
            }
            
            Tensor logits = model->forward(input_tensor);
            
            Tensor last_logits = logits.view({logits.shape[0] * logits.shape[1], logits.shape[2]});
            last_logits = last_logits.view({1, logits.shape[2]});
            
            int next_token;
            if (top_k > 0) {
                next_token = Sampling::sample_top_k(last_logits, top_k);
            } else if (top_p > 0.0f && top_p < 1.0f) {
                next_token = Sampling::sample_top_p(last_logits, top_p);
            } else if (temperature > 0.0f && temperature != 1.0f) {
                next_token = Sampling::sample_temperature(last_logits, temperature);
            } else {
                next_token = Sampling::sample_argmax(last_logits);
            }
            
            generated.push_back(next_token);
            
            if (next_token == 0) break;
        }
        
        return generated;
    }
    
    std::string generate_text(const std::string& prompt_str, int max_new_tokens) {
        std::vector<int> prompt;
        
        for (char c : prompt_str) {
            prompt.push_back(static_cast<int>(static_cast<unsigned char>(c)) % model->vocab_size);
        }
        
        std::vector<int> result = generate(prompt, max_new_tokens);
        
        std::string output;
        for (size_t i = 0; i < result.size(); ++i) {
            if (i < prompt.size()) continue;
            output += static_cast<char>(result[i] % 128);
        }
        
        return output;
    }
    
    void print_generation(const std::vector<int>& tokens) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << tokens[i] << " ";
        }
        std::cout << std::endl;
    }
};

}
}
}
