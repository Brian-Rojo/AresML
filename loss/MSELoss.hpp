#pragma once

#include "../core/Tensor.hpp"
#include "../core/Autograd.hpp"
#include "../utils/Profiler.hpp"
#include <cmath>
#include <vector>

namespace aresml {
namespace loss {

struct MSELossOp : Operation {
    Tensor* pred;
    Tensor* target;
    size_t n;
    
    MSELossOp(Tensor* p, Tensor* t, size_t n_) : pred(p), target(t), n(n_) {}
    
    void backward(Tensor& grad) override {
        if (!pred->grad || !pred->grad->data) {
            pred->grad = std::make_shared<Tensor>(pred->shape);
            pred->grad->zero_();
        }
        
        float scale = 2.0f / static_cast<float>(n);
        if (grad.grad && grad.grad->data) {
            scale *= grad.grad->data[grad.grad->offset];
        }
        
        float* g = pred->grad->data.get() + pred->grad->offset;
        for (size_t i = 0; i < n; ++i) {
            float diff = pred->data[ pred->offset + i] - target->data[target->offset + i];
            g[i] += scale * diff;
        }
    }
    
    std::vector<Tensor*> get_inputs() const override {
        return {pred};
    }

    std::unique_ptr<Operation> clone() const override {
        return std::make_unique<MSELossOp>(pred, target, n);
    }
};

struct MSELoss {
    Tensor forward(const Tensor& pred, const Tensor& target) {
        PROFILE_SCOPE("loss::MSELoss::forward");
        
        size_t n = pred.shape.size();
        
        Tensor loss({1});
        loss.set_requires_grad(true);
        
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            float diff = pred.data[pred.offset + i] - target.data[target.offset + i];
            sum += diff * diff;
        }
        loss.data[0] = sum / static_cast<float>(n);
        
        if (pred.requires_grad) {
            loss.inputs.clear();
            loss.inputs.push_back(const_cast<Tensor*>(&pred));
            loss.op = std::make_unique<MSELossOp>(const_cast<Tensor*>(&pred), const_cast<Tensor*>(&target), n);
        }
        
        return loss;
    }
};

}
}
