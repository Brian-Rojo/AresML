#pragma once

#include "../../core/Tensor.hpp"
#include <cmath>
#include <vector>

namespace aresml {
namespace nn {
namespace transformer {

struct CausalMask {
    static Tensor apply(const Tensor& scores) {
        size_t batch_size = 1;
        size_t num_heads = 1;
        size_t seq_len = 1;
        
        if (scores.shape.n >= 1) batch_size = scores.shape[0];
        if (scores.shape.n >= 2) num_heads = scores.shape[1];
        if (scores.shape.n >= 3) seq_len = scores.shape[2];
        
        Tensor masked(scores.shape, false);
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t j = 0; j < seq_len; ++j) {
                        size_t src_idx = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        size_t dst_idx = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        
                        if (j > i) {
                            masked.data[masked.offset + dst_idx] = -1e9f;
                        } else {
                            masked.data[masked.offset + dst_idx] = scores.data[scores.offset + src_idx];
                        }
                    }
                }
            }
        }
        
        return masked;
    }
    
    static Tensor create(size_t seq_len) {
        Tensor mask({seq_len, seq_len}, false);
        
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j < seq_len; ++j) {
                if (j > i) {
                    mask.data[i * seq_len + j] = -1e9f;
                } else {
                    mask.data[i * seq_len + j] = 0.0f;
                }
            }
        }
        
        return mask;
    }
    
    static Tensor create_batch(size_t batch_size, size_t seq_len) {
        Tensor mask({batch_size, seq_len, seq_len}, false);
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < seq_len; ++j) {
                    size_t idx = b * seq_len * seq_len + i * seq_len + j;
                    if (j > i) {
                        mask.data[idx] = -1e9f;
                    } else {
                        mask.data[idx] = 0.0f;
                    }
                }
            }
        }
        
        return mask;
    }
    
    static Tensor create_causal_mask(size_t seq_len) {
        return create(seq_len);
    }
};

}
}
}
