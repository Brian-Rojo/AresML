#pragma once

#include "../core/Tensor.hpp"
#include <vector>
#include <cmath>

namespace aresml {
namespace optim {

struct SGD {
    float lr;
    float weight_decay = 0.0f;
    
    std::vector<Tensor*> parameters;
    
    SGD(const std::vector<Tensor*>& params, float lr, float weight_decay = 0.0f)
        : lr(lr), weight_decay(weight_decay) {
        for (auto* p : params) {
            if (p && p->requires_grad && p->is_leaf) {
                parameters.push_back(p);
            }
        }
    }
    
    void step() {
        for (auto* p : parameters) {
            if (!p || !p->grad || !p->data) continue;
            
            float* grad = p->grad->data.get() + p->grad->offset;
            float* data = p->data.get() + p->offset;
            size_t n = p->shape.size();
            
            if (weight_decay > 0.0f) {
                for (size_t j = 0; j < n; ++j) {
                    grad[j] += weight_decay * data[j];
                }
            }
            
            for (size_t j = 0; j < n; ++j) {
                data[j] -= lr * grad[j];
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

struct Adam {
    float lr;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.0f;
    
    std::vector<Tensor*> parameters;
    std::vector<Tensor> m;
    std::vector<Tensor> v;
    size_t t = 0;
    
    Adam(const std::vector<Tensor*>& params, float lr, float beta1 = 0.9f, 
         float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f)
        : lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay) {
        for (auto* p : params) {
            if (p && p->requires_grad && p->is_leaf) {
                parameters.push_back(p);
                m.emplace_back(p->shape);
                v.emplace_back(p->shape);
            }
        }
    }
    
    void step() {
        ++t;
        float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(t));
        float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(t));
        
        for (size_t i = 0; i < parameters.size(); ++i) {
            Tensor* p = parameters[i];
            if (!p || !p->grad || !p->data) continue;
            
            float* grad = p->grad->data.get() + p->grad->offset;
            float* data = p->data.get() + p->offset;
            float* mi = m[i].data.get();
            float* vi = v[i].data.get();
            size_t n = p->shape.size();
            
            if (weight_decay > 0.0f) {
                for (size_t j = 0; j < n; ++j) {
                    grad[j] += weight_decay * data[j];
                }
            }
            
            for (size_t j = 0; j < n; ++j) {
                mi[j] = beta1 * mi[j] + (1.0f - beta1) * grad[j];
                vi[j] = beta2 * vi[j] + (1.0f - beta2) * grad[j] * grad[j];
            }
            
            float lr_hat = lr * std::sqrt(bias_correction2) / bias_correction1;
            
            for (size_t j = 0; j < n; ++j) {
                data[j] -= lr_hat * mi[j] / (std::sqrt(vi[j]) + eps);
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
