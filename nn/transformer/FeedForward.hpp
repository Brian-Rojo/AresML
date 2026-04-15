#pragma once

#include "../../core/Tensor.hpp"
#include "../../core/Autograd.hpp"
#include "../Linear.hpp"
#include "../../ops/Ops.hpp"
#include <cmath>
#include <vector>

namespace aresml {
namespace nn {
namespace transformer {

struct FeedForward {
    size_t embed_dim;
    size_t hidden_dim;
    Linear fc1;
    Linear fc2;
    
    FeedForward(size_t embed_dim, size_t hidden_dim = 0)
        : embed_dim(embed_dim), 
          hidden_dim(hidden_dim ? hidden_dim : embed_dim * 4),
          fc1("ffn.fc1", embed_dim, this->hidden_dim, false),
          fc2("ffn.fc2", this->hidden_dim, embed_dim, false) {}
    
    Tensor forward(const Tensor& x) {
        size_t batch_size = 1;
        size_t seq_len = 1;
        
        if (x.shape.n >= 1) batch_size = x.shape[0];
        if (x.shape.n >= 2) seq_len = x.shape[1];
        
        Tensor hidden = fc1.forward(x);
        
        for (size_t i = 0; i < hidden.shape.size(); ++i) {
            hidden.data[hidden.offset + i] = ops::silu_float(hidden.data[hidden.offset + i]);
        }
        
        Tensor output = fc2.forward(hidden);
        
        return output;
    }
};

struct FeedForwardGated {
    size_t embed_dim;
    size_t hidden_dim;
    Linear gate_proj;
    Linear up_proj;
    Linear down_proj;
    
    FeedForwardGated(size_t embed_dim, size_t hidden_dim = 0)
        : embed_dim(embed_dim),
          hidden_dim(hidden_dim ? hidden_dim : embed_dim * 4),
          gate_proj("ffn.gate", embed_dim, this->hidden_dim, false),
          up_proj("ffn.up", embed_dim, this->hidden_dim, false),
          down_proj("ffn.down", this->hidden_dim, embed_dim, false) {}
    
    Tensor forward(const Tensor& x) {
        Tensor gate = gate_proj.forward(x);
        Tensor up = up_proj.forward(x);
        
        for (size_t i = 0; i < gate.shape.size(); ++i) {
            float val = ops::silu_float(gate.data[gate.offset + i]);
            gate.data[gate.offset + i] = val * up.data[up.offset + i];
        }
        
        Tensor output = down_proj.forward(gate);
        
        return output;
    }
};

}
}
}
