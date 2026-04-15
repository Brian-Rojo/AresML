#pragma once

#include "../../core/Tensor.hpp"
#include "Tokenizer.hpp"
#include <vector>
#include <string>
#include <fstream>

namespace aresml {
namespace engine {
namespace gpt {

struct GPTDataset {
    std::vector<std::vector<int>> inputs;
    std::vector<std::vector<int>> targets;
    size_t vocab_size;
    
    GPTDataset() : vocab_size(0) {}
    
    static GPTDataset from_text(const std::string& text, size_t vocab_size, size_t seq_len) {
        std::vector<char> chars;
        for (char c : text) {
            chars.push_back(c);
        }
        
        std::vector<std::vector<int>> token_data;
        
        for (size_t i = 0; i + seq_len < chars.size(); ++i) {
            std::vector<int> inp(seq_len);
            std::vector<int> tgt(seq_len);
            
            for (size_t j = 0; j < seq_len; ++j) {
                inp[j] = static_cast<int>(static_cast<unsigned char>(chars[i + j])) % vocab_size;
                tgt[j] = static_cast<int>(static_cast<unsigned char>(chars[i + j + 1])) % vocab_size;
            }
            
            token_data.push_back(inp);
        }
        
        GPTDataset ds;
        ds.vocab_size = vocab_size;
        
        for (size_t i = 0; i < token_data.size(); ++i) {
            ds.inputs.push_back(token_data[i]);
            
            std::vector<int> shifted(token_data[i].size());
            for (size_t j = 0; j < token_data[i].size() - 1; ++j) {
                shifted[j] = token_data[i][j + 1];
            }
            if (!shifted.empty()) {
                shifted[shifted.size() - 1] = token_data[i][token_data[i].size() - 1];
            }
            ds.targets.push_back(shifted);
        }
        
        return ds;
    }
    
    static GPTDataset simple_dataset(size_t num_samples, size_t seq_len, size_t vocab_size) {
        GPTDataset ds;
        ds.vocab_size = vocab_size;
        
        for (size_t i = 0; i < num_samples; ++i) {
            std::vector<int> inp(seq_len);
            std::vector<int> tgt(seq_len);
            
            for (size_t j = 0; j < seq_len; ++j) {
                inp[j] = (i + j) % vocab_size;
                tgt[j] = (i + j + 1) % vocab_size;
            }
            
            ds.inputs.push_back(inp);
            ds.targets.push_back(tgt);
        }
        
        return ds;
    }
    
    size_t size() const {
        return inputs.size();
    }
    
    std::pair<Tensor, Tensor> get(size_t idx) const {
        Tensor inp(Shape{1, inputs[idx].size()}, false);
        Tensor tgt(Shape{1, targets[idx].size()}, false);
        
        for (size_t i = 0; i < inputs[idx].size(); ++i) {
            inp.data[i] = static_cast<float>(inputs[idx][i]);
            tgt.data[i] = static_cast<float>(targets[idx][i]);
        }
        
        return {inp, tgt};
    }
};

}
}
}
