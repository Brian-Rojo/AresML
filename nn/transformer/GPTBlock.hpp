#pragma once

#include "../../core/Tensor.hpp"
#include "LayerNorm.hpp"
#include "MultiHeadSelfAttention.hpp"
#include "FeedForward.hpp"

namespace aresml {
namespace nn {
namespace transformer {

struct GPTBlock {
    size_t embed_dim;
    size_t num_heads;
    size_t hidden_dim;
    
    LayerNorm ln1;
    LayerNorm ln2;
    MultiHeadSelfAttention attn;
    FeedForward mlp;
    
    GPTBlock(size_t embed_dim, size_t num_heads, size_t hidden_dim = 0)
        : embed_dim(embed_dim), num_heads(num_heads),
          hidden_dim(hidden_dim ? hidden_dim : embed_dim * 4),
          ln1(embed_dim), ln2(embed_dim),
          attn(embed_dim, num_heads, false),
          mlp(embed_dim, this->hidden_dim) {}
    
    Tensor forward(const Tensor& x) {
        size_t batch_size = 1;
        size_t seq_len = 1;
        
        if (x.shape.n >= 1) batch_size = x.shape[0];
        if (x.shape.n >= 2) seq_len = x.shape[1];
        
        Tensor residual = x.clone();
        
        Tensor normed = ln1.forward(x);
        Tensor attn_out = attn.forward(normed, true);
        
        Tensor output = x.clone();
        for (size_t i = 0; i < output.shape.size(); ++i) {
            output.data[output.offset + i] = residual.data[residual.offset + i] + attn_out.data[attn_out.offset + i];
        }
        
        residual = output.clone();
        
        normed = ln2.forward(output);
        Tensor ffn_out = mlp.forward(normed);
        
        for (size_t i = 0; i < output.shape.size(); ++i) {
            output.data[output.offset + i] = residual.data[residual.offset + i] + ffn_out.data[ffn_out.offset + i];
        }
        
        return output;
    }
};

}
}
}
