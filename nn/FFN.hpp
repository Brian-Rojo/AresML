#pragma once

#include "../core/Tensor.hpp"
#include "../core/Autograd.hpp"
#include "Linear.hpp"
#include "../ops/Ops.hpp"
#include <cmath>
#include <vector>

namespace aresml {
namespace nn {

struct FFN {
    size_t hidden_dim;
    size_t embed_dim;
    Linear gate_proj;
    Linear up_proj;
    Linear down_proj;
    
    FFN(size_t embed_dim, size_t hidden_dim = 0) 
        : hidden_dim(hidden_dim ? hidden_dim : (8 * embed_dim / 3)),
          embed_dim(embed_dim),
          gate_proj(embed_dim, this->hidden_dim, false),
          up_proj(embed_dim, this->hidden_dim, false),
          down_proj(this->hidden_dim, embed_dim, false) {}
    
    Tensor forward(const Tensor& x) {
        Tensor input = x;
        
        if (x.shape.n == 3) {
            size_t B = x.shape[0];
            size_t S = x.shape[1];
            input = x.view({B * S, embed_dim});
        } else if (x.shape.n == 2) {
            input = x.view({x.shape[0], embed_dim});
        } else {
            input = x.view({1, embed_dim});
        }
        
        Tensor gate = gate_proj.forward(input);
        Tensor up = up_proj.forward(input);
        
        for (size_t i = 0; i < gate.shape[0] * gate.shape[1]; ++i) {
            gate.data[gate.offset + i] = ops::silu_float(gate.data[gate.offset + i]);
        }
        
        for (size_t i = 0; i < gate.shape[0] * gate.shape[1]; ++i) {
            gate.data[gate.offset + i] *= up.data[up.offset + i];
        }
        
        Tensor out = down_proj.forward(gate);
        
        return out;
    }
};

}
}
