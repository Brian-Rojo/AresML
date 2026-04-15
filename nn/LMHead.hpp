#pragma once

#include "../core/Tensor.hpp"
#include "Linear.hpp"

namespace aresml {
namespace nn {

struct LMHead {
    Linear linear;
    
    LMHead(size_t vocab_size, size_t embed_dim) : linear(embed_dim, vocab_size, false) {}
    
    Tensor forward(const Tensor& x) {
        return linear.forward(x);
    }
};

}
}
