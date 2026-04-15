#pragma once

#include "../../core/Tensor.hpp"
#include "../../core/Autograd.hpp"
#include <cmath>
#include <vector>

namespace aresml {
namespace nn {
namespace transformer {

struct PositionalEncoding {
    size_t max_seq_len;
    size_t embed_dim;
    Tensor pe;
    
    PositionalEncoding(size_t max_seq_len, size_t embed_dim)
        : max_seq_len(max_seq_len), embed_dim(embed_dim) {
        pe = tensor_zeros({max_seq_len, embed_dim}, false);
        
        for (size_t pos = 0; pos < max_seq_len; ++pos) {
            for (size_t i = 0; i < embed_dim; ++i) {
                float angle = pos / std::pow(10000.0f, 2.0f * static_cast<float>(i) / static_cast<float>(embed_dim));
                if (i % 2 == 0) {
                    pe.data[pos * embed_dim + i] = std::sin(angle);
                } else {
                    pe.data[pos * embed_dim + i] = std::cos(angle);
                }
            }
        }
    }
    
    Tensor forward(const Tensor& x) {
        size_t batch_size = 1;
        size_t seq_len = 1;
        
        if (x.shape.n >= 1) batch_size = x.shape[0];
        if (x.shape.n >= 2) seq_len = x.shape[1];
        
        size_t actual_seq_len = std::min(seq_len, max_seq_len);
        
        Tensor output({batch_size, seq_len, embed_dim}, false);
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t pe_row = (s < max_seq_len) ? s : max_seq_len - 1;
                for (size_t e = 0; e < embed_dim; ++e) {
                    size_t x_idx = b * seq_len * embed_dim + s * embed_dim + e;
                    size_t pe_idx = pe_row * embed_dim + e;
                    output.data[x_idx] = x.data[x.offset + x_idx] + pe.data[pe.offset + pe_idx];
                }
            }
        }
        
        return output;
    }
};

struct PositionalAddBackwardOp : Operation {
    Tensor* weight;
    size_t batch_size, seq_len, embed_dim;
    
    PositionalAddBackwardOp(Tensor* w, size_t b, size_t s, size_t e)
        : weight(w), batch_size(b), seq_len(s), embed_dim(e) {}
    
    void backward(Tensor& grad) override {
        if (!grad.grad || !grad.grad->data) return;
        if (!weight || !weight->grad || !weight->grad->data) return;
        
        float* g_out = grad.grad->data.get() + grad.grad->offset;
        float* g_w = weight->grad->data.get() + weight->grad->offset;
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t out_idx = b * seq_len * embed_dim + s * embed_dim;
                size_t w_idx = s * embed_dim;
                for (size_t e = 0; e < embed_dim; ++e) {
                    g_w[w_idx + e] += g_out[out_idx + e];
                }
            }
        }
    }
    
    std::vector<Tensor*> get_inputs() const override {
        return {weight};
    }

    std::unique_ptr<Operation> clone() const override {
        return std::make_unique<PositionalAddBackwardOp>(weight, batch_size, seq_len, embed_dim);
    }
};

struct PositionalEncodingLearned {
    size_t max_seq_len;
    size_t embed_dim;
    Tensor weight;
    
    PositionalEncodingLearned(size_t max_seq_len, size_t embed_dim)
        : max_seq_len(max_seq_len), embed_dim(embed_dim) {
        weight = tensor_randn({max_seq_len, embed_dim}, false);
        weight.set_requires_grad(true);
        weight.is_leaf = true;
        // CRITICAL FIX: Register parameters in global registry
        get_engine().register_parameter(&weight);
    }
    
    Tensor forward(const Tensor& x) {
        size_t batch_size = 1;
        size_t seq_len = 1;
        
        if (x.shape.n >= 1) batch_size = x.shape[0];
        if (x.shape.n >= 2) seq_len = x.shape[1];
        
        size_t actual_seq_len = std::min(seq_len, max_seq_len);
        
        Tensor output({batch_size, seq_len, embed_dim}, x.requires_grad);
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t pe_row = (s < max_seq_len) ? s : max_seq_len - 1;
                for (size_t e = 0; e < embed_dim; ++e) {
                    size_t x_idx = b * seq_len * embed_dim + s * embed_dim + e;
                    size_t w_idx = pe_row * embed_dim + e;
                    output.data[output.offset + x_idx] = x.data[x.offset + x_idx] + weight.data[weight.offset + w_idx];
                }
            }
        }
        
        if (x.requires_grad) {
            output.set_requires_grad(true);
            output.op = std::make_unique<PositionalAddBackwardOp>(&weight, batch_size, seq_len, embed_dim);
            output.inputs.clear();
            output.inputs.push_back(const_cast<Tensor*>(&x));
            output.inputs.push_back(&weight);
        }
        
        return output;
    }
};

}
}
}
