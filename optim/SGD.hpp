#pragma once

#include "../core/Tensor.hpp"
#include "../utils/Profiler.hpp"
#include <vector>

namespace aresml {
namespace optim {

struct SGD {
    float lr;
    float momentum = 0.0f;
    float weight_decay = 0.0f;
    bool nesterov = false;
    
    std::vector<Tensor*> parameters;
    std::vector<Tensor> velocity;
    
    SGD(const std::vector<Tensor*>& params, float lr, float momentum = 0.0f, 
        float weight_decay = 0.0f, bool nesterov = false)
        : lr(lr), momentum(momentum), weight_decay(weight_decay), nesterov(nesterov) {
        for (auto* p : params) {
            if (p && p->requires_grad) {
                parameters.push_back(p);
                if (momentum > 0.0f) {
                    velocity.emplace_back(p->shape);
                }
            }
        }
    }
    
    void step() {
        PROFILE_SCOPE("optim::SGD::step");
        
        for (size_t i = 0; i < parameters.size(); ++i) {
            Tensor* p = parameters[i];
            if (!p || !p->grad || !p->data) continue;
            
            float* grad = p->grad->data.get() + p->grad->offset;
            float* data = p->data.get() + p->offset;
            size_t n = p->shape.size();
            
            if (weight_decay > 0.0f) {
                for (size_t j = 0; j < n; ++j) {
                    grad[j] += weight_decay * data[j];
                }
            }
            
            if (momentum > 0.0f) {
                float* v = velocity[i].data.get();
                float beta = momentum;
                
                for (size_t j = 0; j < n; ++j) {
                    v[j] = beta * v[j] + lr * grad[j];
                }
                
                if (nesterov) {
                    for (size_t j = 0; j < n; ++j) {
                        data[j] -= (1 - beta) * v[j] + beta * (v[j] - beta * velocity[i].data[j]);
                    }
                } else {
                    for (size_t j = 0; j < n; ++j) {
                        data[j] -= v[j];
                    }
                }
            } else {
                for (size_t j = 0; j < n; ++j) {
                    data[j] -= lr * grad[j];
                }
            }
        }
    }
    
    void zero_grad() {
        for (auto* p : parameters) {
            if (p && p->grad) {
                p->grad->zero_();
            }
        }
    }
};

}
}
