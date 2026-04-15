#pragma once

#include "../../core/Tensor.hpp"
#include "../../core/Autograd.hpp"
#include <cmath>
#include <vector>

namespace aresml {
namespace nn {
namespace transformer {

struct LayerNormBackwardOp : Operation {
    Tensor* input;
    Tensor* weight;
    Tensor* bias;
    size_t batch_size;
    size_t seq_len;
    size_t hidden_size;
    float eps;
    
    LayerNormBackwardOp(Tensor* inp, Tensor* w, Tensor* b, size_t bsz, size_t s, size_t h, float e)
        : input(inp), weight(w), bias(b), batch_size(bsz), seq_len(s), hidden_size(h), eps(e) {}
    
    void backward(Tensor& grad) override {
        if (!grad.grad || !grad.grad->data) return;
        
        const float* g_out = grad.grad->data.get() + grad.grad->offset;
        
        if (weight && weight->grad && weight->grad->data) {
            float* g_w = weight->grad->data.get() + weight->grad->offset;
            const float* in_data = input->data.get() + input->offset;
            
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t s = 0; s < seq_len; ++s) {
                    float mean = 0.0f;
                    for (size_t h = 0; h < hidden_size; ++h) {
                        size_t idx = b * seq_len * hidden_size + s * hidden_size + h;
                        mean += in_data[idx];
                    }
                    mean /= static_cast<float>(hidden_size);
                    
                    float var = 0.0f;
                    for (size_t h = 0; h < hidden_size; ++h) {
                        size_t idx = b * seq_len * hidden_size + s * hidden_size + h;
                        float diff = in_data[idx] - mean;
                        var += diff * diff;
                    }
                    var /= static_cast<float>(hidden_size);
                    float std = std::sqrt(var + eps);
                    
                    for (size_t h = 0; h < hidden_size; ++h) {
                        size_t idx = b * seq_len * hidden_size + s * hidden_size + h;
                        float normalized = (in_data[idx] - mean) / std;
                        g_w[h] += g_out[idx] * normalized;
                    }
                }
            }
        }
        
        if (bias && bias->grad && bias->grad->data) {
            float* g_b = bias->grad->data.get() + bias->grad->offset;
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t s = 0; s < seq_len; ++s) {
                    for (size_t h = 0; h < hidden_size; ++h) {
                        size_t idx = b * seq_len * hidden_size + s * hidden_size + h;
                        g_b[h] += g_out[idx];
                    }
                }
            }
        }
        
        if (input && input->grad && input->grad->data) {
            float* g_in = input->grad->data.get() + input->grad->offset;
            const float* in_data = input->data.get() + input->offset;
            
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t s = 0; s < seq_len; ++s) {
                    float mean = 0.0f;
                    for (size_t h = 0; h < hidden_size; ++h) {
                        size_t idx = b * seq_len * hidden_size + s * hidden_size + h;
                        mean += in_data[idx];
                    }
                    mean /= static_cast<float>(hidden_size);
                    
                    float var = 0.0f;
                    for (size_t h = 0; h < hidden_size; ++h) {
                        size_t idx = b * seq_len * hidden_size + s * hidden_size + h;
                        float diff = in_data[idx] - mean;
                        var += diff * diff;
                    }
                    var /= static_cast<float>(hidden_size);
                    float std = std::sqrt(var + eps);
                    
                    const float* w_data = weight->data.get() + weight->offset;
                    float w_sum = 0.0f;
                    for (size_t h = 0; h < hidden_size; ++h) {
                        w_sum += w_data[h];
                    }
                    
                    for (size_t h = 0; h < hidden_size; ++h) {
                        size_t idx = b * seq_len * hidden_size + s * hidden_size + h;
                        float normalized = (in_data[idx] - mean) / std;
                        float gw = w_data[h];
                        
                        g_in[idx] += gw * g_out[idx] / std;
                        g_in[idx] -= gw * (g_out[idx] - w_sum * g_out[idx] / static_cast<float>(hidden_size)) / std;
                    }
                }
            }
        }
    }
    
    std::vector<Tensor*> get_inputs() const override {
        return {input, weight, bias};
    }

    std::unique_ptr<Operation> clone() const override {
        return std::make_unique<LayerNormBackwardOp>(input, weight, bias, batch_size, seq_len, hidden_size, eps);
    }
};

struct LayerNorm {
    size_t hidden_size;
    float eps;
    Tensor weight;
    Tensor bias;
    
    LayerNorm(size_t hidden_size, float eps = 1e-5f)
        : hidden_size(hidden_size), eps(eps) {
        weight = tensor_ones({hidden_size}, false);
        weight.set_requires_grad(true);
        weight.is_leaf = true;

        bias = tensor_zeros({hidden_size}, false);
        bias.set_requires_grad(true);
        bias.is_leaf = true;

        // CRITICAL FIX: Register parameters in global registry
        get_engine().register_parameter(&weight);
        get_engine().register_parameter(&bias);
    }
    
    Tensor forward(const Tensor& x) {
        size_t batch_size = 1;
        size_t seq_len = 1;
        
        if (x.shape.n >= 1) batch_size = x.shape[0];
        if (x.shape.n >= 2) seq_len = x.shape[1];
        
        size_t actual_hidden = hidden_size;
        if (x.shape.n >= 3) actual_hidden = x.shape[2];
        
        Tensor output({batch_size, seq_len, actual_hidden}, x.requires_grad);
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                float mean = 0.0f;
                for (size_t h = 0; h < actual_hidden; ++h) {
                    size_t idx = b * seq_len * actual_hidden + s * actual_hidden + h;
                    mean += x.data[x.offset + idx];
                }
                mean /= static_cast<float>(actual_hidden);
                
                float var = 0.0f;
                for (size_t h = 0; h < actual_hidden; ++h) {
                    size_t idx = b * seq_len * actual_hidden + s * actual_hidden + h;
                    float diff = x.data[x.offset + idx] - mean;
                    var += diff * diff;
                }
                var /= static_cast<float>(actual_hidden);
                float std = std::sqrt(var + eps);
                
                for (size_t h = 0; h < actual_hidden; ++h) {
                    size_t idx = b * seq_len * actual_hidden + s * actual_hidden + h;
                    float normalized = (x.data[x.offset + idx] - mean) / std;
                    float w = weight.data[weight.offset + h];
                    float b_val = bias.data[bias.offset + h];
                    output.data[output.offset + idx] = normalized * w + b_val;
                }
            }
        }
        
        if (x.requires_grad) {
            output.set_requires_grad(true);
            output.inputs.clear();
            output.inputs.push_back(const_cast<Tensor*>(&x));
            output.inputs.push_back(&weight);
            output.inputs.push_back(&bias);
            output.op = std::make_unique<LayerNormBackwardOp>(
                const_cast<Tensor*>(&x), &weight, &bias, batch_size, seq_len, actual_hidden, eps);
        }
        
        return output;
    }
};

}
}
}
