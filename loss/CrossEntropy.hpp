#pragma once

#include "../core/Tensor.hpp"
#include "../core/Autograd.hpp"
#include "../backend_cpu/Softmax.hpp"
#include <cmath>
#include <vector>

namespace aresml {
namespace loss {

struct CrossEntropyOp : Operation {
    Tensor* logits;  // Guardar logits en lugar de log_probs (variable local)
    std::vector<size_t> targets_data;
    size_t batch_size, seq_len, vocab_size;
    
    CrossEntropyOp(Tensor* lg, const Tensor& targets, size_t b, size_t s, size_t v)
        : logits(lg), batch_size(b), seq_len(s), vocab_size(v) {
        targets_data.resize(batch_size * seq_len);
        for (size_t i = 0; i < batch_size * seq_len; ++i) {
            targets_data[i] = static_cast<size_t>(targets.data[targets.offset + i]);
        }
    }
    
    void backward(Tensor& grad) override {
        // CRITICAL: Validar TODAS las condiciones
        if (!grad.grad || !grad.grad->data) {
            std::cout << "[CE backward] grad.grad is null" << std::endl;
            return;
        }
        if (!logits) {
            std::cout << "[CE backward] logits is null" << std::endl;
            return;
        }
        if (!logits->grad) {
            std::cout << "[CE backward] logits grad is null - CREATING IT" << std::endl;
            logits->grad = std::make_shared<Tensor>(logits->shape);
            logits->grad->zero_();
        }
        if (!logits->grad->data) {
            std::cout << "[CE backward] logits grad data is null" << std::endl;
            return;
        }
        
        std::cout << "[CE backward] Computing gradients for logits shape=" << logits->shape.size() << std::endl;
        
        // Verificar shapes
        if (logits->shape.size() != batch_size * seq_len * vocab_size) {
            return;  // Shape mismatch
        }
        
        float* g_logits = logits->grad->data.get() + logits->grad->offset;
        const float* g_out = grad.grad->data.get() + grad.grad->offset;
        
        const float scale = g_out[0] / static_cast<float>(batch_size * seq_len);
        
        // Compute softmax(logits) en el backward
        // Luego aplicar regla de cadena desde CE a logits
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t target_idx_in_seq = b * seq_len + s;
                size_t target = targets_data[target_idx_in_seq];
                
                if (target >= vocab_size) {
                    continue;
                }
                
                // Para cada output en esta posición
                for (size_t v = 0; v < vocab_size; ++v) {
                    size_t idx = b * seq_len * vocab_size + s * vocab_size + v;
                    
                    if (idx >= logits->shape.size()) {
                        continue;
                    }
                    
                    // grad_logits[v] = scale * (softmax[v] - delta[v, target])
                    // Necesitamos computar softmax[v]
                    const float* logits_data = logits->data.get() + logits->offset;
                    
                    // Max-subtraction para estabilidad numérica
                    float max_logit = -1e38f;
                    for (size_t vv = 0; vv < vocab_size; ++vv) {
                        size_t idx_vv = b * seq_len * vocab_size + s * vocab_size + vv;
                        max_logit = std::max(max_logit, logits_data[idx_vv]);
                    }
                    
                    // Computar softmax[v]
                    float exp_logit = std::exp(logits_data[idx] - max_logit);
                    float sum_exp = 0.0f;
                    for (size_t vv = 0; vv < vocab_size; ++vv) {
                        size_t idx_vv = b * seq_len * vocab_size + s * vocab_size + vv;
                        sum_exp += std::exp(logits_data[idx_vv] - max_logit);
                    }
                    
                    float softmax_v = exp_logit / (sum_exp + EPSILON);
                    
                    // Aplicar regla CE: grad = softmax - one_hot
                    float grad_contrib = softmax_v;
                    if (v == target) {
                        grad_contrib -= 1.0f;
                    }
                    
                    g_logits[idx] += scale * grad_contrib;
                }
            }
        }
    }
    
    std::vector<Tensor*> get_inputs() const override {
        return {logits};
    }

    std::unique_ptr<Operation> clone() const override {
        auto cloned = std::make_unique<CrossEntropyOp>(logits, Tensor(), batch_size, seq_len, vocab_size);
        cloned->targets_data = targets_data;
        return cloned;
    }
};

struct CrossEntropyLoss {
    bool reduction_mean = true;
    
    CrossEntropyLoss(bool reduction_mean = true) : reduction_mean(reduction_mean) {}
    
    Tensor forward(const Tensor& logits, const Tensor& targets) {
        size_t batch_size = logits.shape[0];
        size_t vocab_size = logits.shape[logits.shape.n - 1];
        
        size_t seq_len = 1;
        if (logits.shape.n > 2) {
            seq_len = logits.shape[1];
        }
        
        Tensor log_probs = backend_cpu::log_softmax(logits, -1);
        
        Tensor loss({batch_size, seq_len});
        loss.set_requires_grad(true);
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t target = static_cast<size_t>(targets.data[targets.offset + b * seq_len + s]);
                float log_prob = log_probs.data[log_probs.offset + b * seq_len * vocab_size + s * vocab_size + target];
                loss.data[loss.offset + b * seq_len + s] = -log_prob;
            }
        }
        
        if (reduction_mean) {
            float total_loss = loss.sum();
            float numel = static_cast<float>(batch_size * seq_len);
            Tensor loss_scalar({1});
            loss_scalar.set_requires_grad(true);
            loss_scalar.data[0] = total_loss / numel;
            
            if (logits.requires_grad) {
                loss_scalar.op = std::make_unique<CrossEntropyOp>(
                    const_cast<Tensor*>(&logits), targets,
                    batch_size, seq_len, vocab_size);
                loss_scalar.inputs.clear();
                loss_scalar.inputs.push_back(const_cast<Tensor*>(&logits));
            }
            
            return loss_scalar;
        }
        
        return loss;
    }
};

}
}
