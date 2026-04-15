#pragma once

#include "../core/Tensor.hpp"
#include "RMSNorm.hpp"
#include "Attention.hpp"
#include "FFN.hpp"

namespace aresml {
namespace nn {

struct TransformerBlock {
    size_t embed_dim;
    size_t num_heads;
    RMSNorm attn_norm;
    RMSNorm ffn_norm;
    Attention attention;
    FFN ffn;
    
    TransformerBlock(size_t embed_dim, size_t num_heads)
        : embed_dim(embed_dim), num_heads(num_heads),
          attn_norm(embed_dim), ffn_norm(embed_dim),
          attention(embed_dim, num_heads), ffn(embed_dim) {}
    
    Tensor forward(const Tensor& x) {
        Tensor output = x.clone();
        output.set_requires_grad(x.requires_grad);
        
        Tensor residual = output;
        
        Tensor normed = attn_norm.forward(output);
        Tensor attn_out = attention.forward(normed);
        
        for (size_t i = 0; i < output.shape[0] * output.shape[1] * output.shape[2]; ++i) {
            output.data[output.offset + i] = residual.data[residual.offset + i] + attn_out.data[attn_out.offset + i];
        }
        
        residual = output;
        
        normed = ffn_norm.forward(output);
        Tensor ffn_out = ffn.forward(normed);
        
        for (size_t i = 0; i < output.shape[0] * output.shape[1] * output.shape[2]; ++i) {
            output.data[output.offset + i] = residual.data[residual.offset + i] + ffn_out.data[ffn_out.offset + i];
        }
        
        return output;
    }
};

}
}
