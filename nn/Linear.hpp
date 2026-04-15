#pragma once

#include "../core/Tensor.hpp"
#include "../core/Autograd.hpp"
#include "../core/Checkpoint.hpp"
#include "../core/GraphCompiler.hpp"
#include "../backend_cpu/Matmul.hpp"
#include "../utils/Profiler.hpp"
#include <cmath>
#include <vector>

namespace aresml {
namespace nn {

// ============================================================================
// LinearBackwardOp
// ============================================================================
struct LinearBackwardOp : Operation {
    Tensor* input;
    Tensor* weight;
    Tensor* bias;
    size_t B, K, O;

    LinearBackwardOp(Tensor* inp, Tensor* w, Tensor* b, size_t bsz, size_t k, size_t o)
        : input(inp), weight(w), bias(b), B(bsz), K(k), O(o) {}

    void backward(Tensor& grad) override {
        if (!grad.grad || !grad.grad->data) return;

        float* grad_out = grad.grad->data.get() + grad.grad->offset;

        // Gradient w.r.t. weight: dL/dW = X^T @ dL/dY
        if (weight && weight->grad && weight->grad->data) {
            float* gw = weight->grad->data.get() + weight->grad->offset;
            const float* inp_data = input->data.get() + input->offset;
            for (size_t k = 0; k < K; ++k) {
                for (size_t o = 0; o < O; ++o) {
                    float sum = 0.0f;
                    for (size_t b = 0; b < B; ++b) {
                        sum += inp_data[b * K + k] * grad_out[b * O + o];
                    }
                    gw[k * O + o] += sum;
                }
            }
        }

        // Gradient w.r.t. bias: dL/db = sum(dL/dY, axis=0)
        if (bias && bias->grad && bias->grad->data && bias->data) {
            float* gb = bias->grad->data.get() + bias->grad->offset;
            for (size_t o = 0; o < O; ++o) {
                float sum = 0.0f;
                for (size_t b = 0; b < B; ++b) {
                    sum += grad_out[b * O + o];
                }
                gb[o] += sum;
            }
        }

        // Gradient w.r.t. input: dL/dX = dL/dY @ W^T
        if (input && input->grad && input->grad->data) {
            float* g_in = input->grad->data.get() + input->grad->offset;
            const float* w_data = weight->data.get() + weight->offset;
            for (size_t b = 0; b < B; ++b) {
                for (size_t k = 0; k < K; ++k) {
                    float sum = 0.0f;
                    for (size_t o = 0; o < O; ++o) {
                        sum += grad_out[b * O + o] * w_data[k * O + o];
                    }
                    g_in[b * K + k] += sum;
                }
            }
        }
    }

    std::vector<Tensor*> get_inputs() const override {
        std::vector<Tensor*> result;
        if (input) result.push_back(input);
        if (weight) result.push_back(weight);
        if (bias && bias->data) result.push_back(bias);
        return result;
    }

    std::unique_ptr<Operation> clone() const override {
        return std::make_unique<LinearBackwardOp>(input, weight, bias, B, K, O);
    }
};

// ============================================================================
// Linear layer - FIXED: registers parameters in global registry
// ============================================================================
struct Linear : public Module {
    Tensor weight;
    Tensor bias;
    std::string name_prefix;
    bool use_bias;

    Linear(size_t in_features, size_t out_features, bool use_bias_ = true)
        : name_prefix("linear"), use_bias(use_bias_) {
        
        // Initialize weight with Kaiming normal
        weight = tensor_randn({in_features, out_features}, false);
        weight.set_requires_grad(true);
        weight.is_leaf = true;

        float scale = std::sqrt(2.0f / static_cast<float>(in_features));
        for (size_t i = 0; i < weight.shape.size(); ++i) {
            weight.data[i] *= scale;
        }

        if (use_bias) {
            bias = tensor_zeros({out_features}, false);
            bias.set_requires_grad(true);
            bias.is_leaf = true;
        }

        // CRITICAL FIX: Register parameters in global registry
        get_engine().register_parameter(&weight);
        if (use_bias) {
            get_engine().register_parameter(&bias);
        }
    }

    Linear(const std::string& name, size_t in_features, size_t out_features, bool use_bias_ = true)
        : name_prefix(name), use_bias(use_bias_) {
        
        weight = tensor_randn({in_features, out_features}, false);
        weight.set_requires_grad(true);
        weight.is_leaf = true;

        float scale = std::sqrt(2.0f / static_cast<float>(in_features));
        for (size_t i = 0; i < weight.shape.size(); ++i) {
            weight.data[i] *= scale;
        }

        if (use_bias) {
            bias = tensor_zeros({out_features}, false);
            bias.set_requires_grad(true);
            bias.is_leaf = true;
        }

        // CRITICAL FIX: Register parameters in global registry
        get_engine().register_parameter(&weight);
        if (use_bias) {
            get_engine().register_parameter(&bias);
        }
    }

    std::string get_full_name(const std::string& param) const {
        return name_prefix + "." + param;
    }

    std::unordered_map<std::string, Tensor*> state_dict() override {
        std::unordered_map<std::string, Tensor*> dict;
        dict[get_full_name("weight")] = &weight;
        if (use_bias && bias.data) {
            dict[get_full_name("bias")] = &bias;
        }
        return dict;
    }

    void load_state_dict(const std::unordered_map<std::string, Tensor*>& state) override {
        auto it = state.find(get_full_name("weight"));
        if (it != state.end() && it->second && it->second->data) {
            std::memcpy(weight.data.get(), it->second->data.get(), weight.shape.size() * sizeof(float));
        }

        if (use_bias && bias.data) {
            auto it_bias = state.find(get_full_name("bias"));
            if (it_bias != state.end() && it_bias->second && it_bias->second->data) {
                std::memcpy(bias.data.get(), it_bias->second->data.get(), bias.shape.size() * sizeof(float));
            }
        }
    }

    Tensor forward(const Tensor& x) {
        PROFILE_SCOPE("nn::Linear::forward");

        const Tensor* input_ptr = &x;
        Tensor reshaped_view;

        if (x.shape.n == 3) {
            // For 3D input (batch, seq, features), reshape to (batch*seq, features)
            reshaped_view = x.view({x.shape[0] * x.shape[1], x.shape[2]});
            input_ptr = &reshaped_view;
        }

        Tensor out({input_ptr->shape[0], weight.shape[1]}, x.requires_grad);

        // FUSION: matmul + bias in single operation
        if (use_bias && bias.data) {
            backend_cpu::matmul_add_bias(*input_ptr, weight, bias, out);
        } else {
            backend_cpu::matmul(*input_ptr, weight, out);
        }

        if (GraphRecorder::capture_enabled) {
            GraphRecorder::record_linear(const_cast<Tensor*>(&x), &weight, &out, name_prefix);
        }

        if (x.requires_grad) {
            out.set_requires_grad(true);
            out.op = std::make_unique<LinearBackwardOp>(
                const_cast<Tensor*>(input_ptr), &weight, 
                use_bias ? &bias : nullptr,
                input_ptr->shape[0], input_ptr->shape[1], weight.shape[1]);
            
            // FIX: Set inputs correctly so backward can traverse the graph
            out.inputs.clear();
            out.inputs.push_back(&weight);
            out.inputs.push_back(const_cast<Tensor*>(input_ptr));
            if (use_bias && bias.data) {
                out.inputs.push_back(&bias);
            }
        }

        return out;
    }
};

} // namespace nn
} // namespace aresml
