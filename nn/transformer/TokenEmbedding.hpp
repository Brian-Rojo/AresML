#pragma once

#include "../../core/Tensor.hpp"
#include "../../core/Autograd.hpp"
#include <cmath>
#include <vector>
#include <random>

namespace aresml {
namespace nn {
namespace transformer {

// ============================================================================
// EmbeddingBackwardOp
// ============================================================================
struct EmbeddingBackwardOp : Operation {
    Tensor* weight;
    Tensor* input_ids;
    std::vector<size_t> token_ids;
    size_t batch_size;
    size_t seq_len;
    size_t vocab_size;
    size_t embed_dim;

    EmbeddingBackwardOp(Tensor* w, Tensor* inp_ids, size_t b, size_t s, size_t v, size_t e)
        : weight(w), input_ids(inp_ids), batch_size(b), seq_len(s), vocab_size(v), embed_dim(e) {
        token_ids.resize(batch_size * seq_len);
        const int* ids = reinterpret_cast<int*>(inp_ids->data.get());
        for (size_t i = 0; i < batch_size * seq_len; ++i) {
            token_ids[i] = static_cast<size_t>(ids[i]);
        }
    }

    void backward(Tensor& grad) override {
        if (!grad.grad || !grad.grad->data) return;

        float* g_out = grad.grad->data.get() + grad.grad->offset;

        // Scatter gradients from output back to embedding weight rows
        if (weight && weight->grad && weight->grad->data) {
            float* g_w = weight->grad->data.get() + weight->grad->offset;
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t s = 0; s < seq_len; ++s) {
                    size_t token_id = token_ids[b * seq_len + s] % vocab_size;
                    for (size_t e = 0; e < embed_dim; ++e) {
                        size_t grad_idx = b * seq_len * embed_dim + s * embed_dim + e;
                        size_t weight_idx = token_id * embed_dim + e;
                        g_w[weight_idx] += g_out[grad_idx];
                    }
                }
            }
        }
    }

    std::vector<Tensor*> get_inputs() const override {
        std::vector<Tensor*> result;
        if (weight) result.push_back(weight);
        if (input_ids) result.push_back(input_ids);
        return result;
    }

    std::unique_ptr<Operation> clone() const override {
        auto cloned = std::make_unique<EmbeddingBackwardOp>(weight, input_ids, batch_size, seq_len, vocab_size, embed_dim);
        cloned->token_ids = token_ids;
        return cloned;
    }
};

// ============================================================================
// TokenEmbedding - FIXED: registers parameters in global registry
// ============================================================================
struct TokenEmbedding {
    size_t vocab_size;
    size_t embed_dim;
    Tensor weight;

    TokenEmbedding(size_t vocab_size, size_t embed_dim)
        : vocab_size(vocab_size), embed_dim(embed_dim) {
        weight = tensor_randn({vocab_size, embed_dim}, false);
        weight.set_requires_grad(true);
        weight.is_leaf = true;
        
        // CRITICAL FIX: Register weight in global parameter registry
        get_engine().register_parameter(&weight);
    }

    TokenEmbedding(const std::string& name, size_t vocab_size, size_t embed_dim)
        : vocab_size(vocab_size), embed_dim(embed_dim) {
        weight = tensor_randn({vocab_size, embed_dim}, false);
        weight.set_requires_grad(true);
        weight.is_leaf = true;
        
        // CRITICAL FIX: Register weight in global parameter registry
        get_engine().register_parameter(&weight);
    }

    Tensor forward(const Tensor& token_ids) {
        size_t batch_size = 1;
        size_t seq_len = 1;

        if (token_ids.shape.n >= 1) batch_size = token_ids.shape[0];
        if (token_ids.shape.n >= 2) seq_len = token_ids.shape[1];

        Tensor output({batch_size, seq_len, embed_dim}, token_ids.requires_grad);

        const int* ids = reinterpret_cast<int*>(token_ids.data.get());

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t token_id = 0;
                if (b * seq_len + s < token_ids.shape.size()) {
                    token_id = static_cast<size_t>(ids[b * seq_len + s]);
                    token_id = std::min(token_id, vocab_size - 1);
                }

                for (size_t e = 0; e < embed_dim; ++e) {
                    size_t src_idx = token_id * embed_dim + e;
                    size_t dst_idx = b * seq_len * embed_dim + s * embed_dim + e;
                    output.data[output.offset + dst_idx] = weight.data[weight.offset + src_idx];
                }
            }
        }

        if (weight.requires_grad) {
            output.set_requires_grad(true);
            output.op = std::make_unique<EmbeddingBackwardOp>(&weight, const_cast<Tensor*>(&token_ids), 
                                                               batch_size, seq_len, vocab_size, embed_dim);
            output.inputs.clear();
            output.inputs.push_back(&weight);
        }

        return output;
    }
};

} // namespace transformer
} // namespace nn
} // namespace aresml
