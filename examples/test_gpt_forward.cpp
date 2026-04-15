#include "../nn/transformer/GPTModel.hpp"
#include "../core/Autograd.hpp"
#include <iostream>
#include <cassert>

using namespace aresml;
using namespace aresml::nn::transformer;

int main() {
    std::cout << "=== GPT Forward Test ===" << std::endl;
    
    size_t vocab_size = 256;
    size_t embed_dim = 64;
    size_t num_heads = 2;
    size_t num_layers = 1;
    size_t max_seq_len = 16;
    size_t batch_size = 1;
    size_t seq_len = 8;
    
    GPTModel model(vocab_size, embed_dim, num_heads, num_layers, max_seq_len);
    
    Tensor input_ids(Shape{batch_size, seq_len}, false);
    for (size_t i = 0; i < batch_size * seq_len; ++i) {
        input_ids.data[i] = static_cast<float>(i % vocab_size);
    }
    
    std::cout << "Input shape: " << batch_size << " x " << seq_len << std::endl;
    
    Tensor logits = model.forward(input_ids);
    
    std::cout << "Output shape: " << logits.shape[0] << " x " << logits.shape[1];
    if (logits.shape.n >= 2) std::cout << " x " << logits.shape[2];
    std::cout << std::endl;
    
    if (logits.shape.n == 2) {
        assert(logits.shape[0] == batch_size * seq_len);
    } else {
        assert(logits.shape[0] == batch_size);
    }
    
    float max_val = -1e9f;
    for (size_t i = 0; i < logits.shape.size(); ++i) {
        max_val = std::max(max_val, logits.data[logits.offset + i]);
    }
    std::cout << "Max logits value: " << max_val << std::endl;
    
    std::cout << "=== GPT Forward Test PASSED ===" << std::endl;
    
    return 0;
}
