#pragma once

#include "../core/Tensor.hpp"
#include "../core/Autograd.hpp"
#include <cmath>
#include <vector>

namespace aresml {
namespace nn {

struct RMSNormBackwardOp : Operation {
    Tensor* input;
    Tensor* weight;
    size_t batch, seq_len, dim;
    float eps;
    
    RMSNormBackwardOp(Tensor* inp, Tensor* w, size_t b, size_t s, size_t d, float e)
        : input(inp), weight(w), batch(b), seq_len(s), dim(d), eps(e) {}
    
    void backward(Tensor& grad) override {
        if (!grad.grad || !grad.grad->data) return;
        
        const float* g_out = grad.grad->data.get() + grad.grad->offset;
        
        // grad_weight = sum(grad_out * normalized_input)
        if (weight && weight->grad && weight->grad->data) {
            float* g_w = weight->grad->data.get() + weight->grad->offset;
            const float* in_data = input->data.get() + input->offset;
            
            for (size_t d_idx = 0; d_idx < dim; ++d_idx) {
                float sum = 0.0f;
                for (size_t b_idx = 0; b_idx < batch; ++b_idx) {
                    for (size_t s_idx = 0; s_idx < seq_len; ++s_idx) {
                        size_t idx = b_idx * seq_len * dim + s_idx * dim + d_idx;
                        float rms = 0.0f;
                        for (size_t k = 0; k < dim; ++k) {
                            float v = in_data[b_idx * seq_len * dim + s_idx * dim + k];
                            rms += v * v;
                        }
                        rms = std::sqrt(rms / static_cast<float>(dim) + eps);
                        float norm = in_data[idx] / rms;
                        sum += g_out[idx] * norm;
                    }
                }
                g_w[d_idx] += sum;
            }
        }
        
        // grad_input = grad_out * weight / rms * (1 - x^2 / (dim * rms^2))
        if (input && input->grad && input->grad->data) {
            float* g_in = input->grad->data.get() + input->grad->offset;
            const float* in_data = input->data.get() + input->offset;
            
            for (size_t b_idx = 0; b_idx < batch; ++b_idx) {
                for (size_t s_idx = 0; s_idx < seq_len; ++s_idx) {
                    // Compute RMS for this position
                    float sum_sq = 0.0f;
                    for (size_t k = 0; k < dim; ++k) {
                        float v = in_data[b_idx * seq_len * dim + s_idx * dim + k];
                        sum_sq += v * v;
                    }
                    float rms = std::sqrt(sum_sq / static_cast<float>(dim) + eps);
                    float rms_cubed = rms * rms * rms;
                    
                    // Compute normalized input
                    std::vector<float> norm(dim);
                    for (size_t d_idx = 0; d_idx < dim; ++d_idx) {
                        size_t idx = b_idx * seq_len * dim + s_idx * dim + d_idx;
                        norm[d_idx] = in_data[idx] / rms;
                    }
                    
                    // Compute weight
                    const float* w_data = weight->data.get() + weight->offset;
                    
                    // Backprop
                    for (size_t d_idx = 0; d_idx < dim; ++d_idx) {
                        size_t idx = b_idx * seq_len * dim + s_idx * dim + d_idx;
                        float x_sq = in_data[idx] * in_data[idx];
                        float factor = 1.0f - x_sq / (dim * rms * rms);
                        g_in[idx] += g_out[idx] * w_data[d_idx] / rms * factor;
                    }
                }
            }
        }
    }
    
    std::vector<Tensor*> get_inputs() const override {
        return {input, weight};
    }

    std::unique_ptr<Operation> clone() const override {
        return std::make_unique<RMSNormBackwardOp>(input, weight, batch, seq_len, dim, eps);
    }
};

struct RMSNorm {
    float eps = 1e-5f;
    size_t hidden_size;
    Tensor weight;
    
    RMSNorm(size_t hidden_size, float eps = 1e-5f) : hidden_size(hidden_size), eps(eps) {
        weight = tensor_ones({hidden_size}, false);
        weight.set_requires_grad(true);
        weight.is_leaf = true;
        // CRITICAL FIX: Register parameters in global registry
        get_engine().register_parameter(&weight);
    }
    
    Tensor forward(const Tensor& x) {
        size_t batch = 1;
        size_t seq_len = 1;
        size_t dim = hidden_size;
        
        const Tensor* input_ptr = &x;
        
        if (x.shape.n == 3) {
            batch = x.shape[0];
            seq_len = x.shape[1];
            dim = x.shape[2];
        } else if (x.shape.n == 2) {
            batch = x.shape[0];
            dim = x.shape[1];
        } else {
            batch = 1;
            dim = x.shape[0];
        }
        
        Tensor out({batch, seq_len, hidden_size}, x.requires_grad);
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                float sum_sq = 0.0f;
                for (size_t d = 0; d < dim; ++d) {
                    float v = input_ptr->data[input_ptr->offset + b * seq_len * dim + s * dim + d];
                    sum_sq += v * v;
                }
                float rms = std::sqrt(sum_sq / static_cast<float>(dim) + eps);
                
                for (size_t d = 0; d < dim; ++d) {
                    float v = input_ptr->data[input_ptr->offset + b * seq_len * dim + s * dim + d];
                    float norm = v / rms;
                    float w = weight.data[weight.offset + d];
                    out.data[out.offset + b * seq_len * dim + s * dim + d] = norm * w;
                }
            }
        }
        
        if (x.requires_grad) {
            out.set_requires_grad(true);
            out.op = std::make_unique<RMSNormBackwardOp>(
                const_cast<Tensor*>(&x), &weight, batch, seq_len, dim, eps);
            out.inputs.clear();
            out.inputs.push_back(const_cast<Tensor*>(&x));
            out.inputs.push_back(&weight);
        }
        
        return out;
    }
};

}
}
