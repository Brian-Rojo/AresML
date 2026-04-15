#pragma once

#include "../core/Tensor.hpp"
#include "../core/Autograd.hpp"
#include "../core/Checkpoint.hpp"
#include "../backend_cpu/Softmax.hpp"
#include "Linear.hpp"
#include <cmath>
#include <vector>

namespace aresml {
namespace nn {

struct AttentionBackwardOp : Operation {
    Tensor* input;
    Tensor* q_weight;
    Tensor* k_weight;
    Tensor* v_weight;
    Tensor* out_weight;
    
    size_t batch;
    size_t seq_len;
    size_t embed_dim;
    size_t num_heads;
    size_t head_dim;
    
    AttentionBackwardOp(Tensor* inp, Tensor* qw, Tensor* kw, Tensor* vw, Tensor* ow,
                     size_t b, size_t s, size_t e, size_t h, size_t hd)
        : input(inp), q_weight(qw), k_weight(kw), v_weight(vw), out_weight(ow)
        , batch(b), seq_len(s), embed_dim(e), num_heads(h), head_dim(hd) {}
    
    inline float fast_relu(float x) {
        return x > 0.0f ? x : 0.0f;
    }
    
    void backward(Tensor& grad) override {
        if (!grad.grad || !grad.grad->data) return;
        
        const float* g_out = grad.grad->data.get() + grad.grad->offset;
        
        size_t B = batch * seq_len;
        
        Tensor Q = Tensor({B, embed_dim}, false);
        Tensor K = Tensor({B, embed_dim}, false);
        Tensor V = Tensor({B, embed_dim}, false);
        
        if (q_weight && q_weight->data && input && input->data) {
            const float* inp = input->data.get() + input->offset;
            const float* w = q_weight->data.get();
            for (size_t i = 0; i < B; ++i) {
                for (size_t j = 0; j < embed_dim; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < embed_dim; ++k) {
                        sum += inp[i * embed_dim + k] * w[k * embed_dim + j];
                    }
                    Q.data[i * embed_dim + j] = sum;
                }
            }
        }
        
        if (k_weight && k_weight->data && input && input->data) {
            const float* inp = input->data.get() + input->offset;
            const float* w = k_weight->data.get();
            for (size_t i = 0; i < B; ++i) {
                for (size_t j = 0; j < embed_dim; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < embed_dim; ++k) {
                        sum += inp[i * embed_dim + k] * w[k * embed_dim + j];
                    }
                    K.data[i * embed_dim + j] = sum;
                }
            }
        }
        
        if (v_weight && v_weight->data && input && input->data) {
            const float* inp = input->data.get() + input->offset;
            const float* w = v_weight->data.get();
            for (size_t i = 0; i < B; ++i) {
                for (size_t j = 0; j < embed_dim; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < embed_dim; ++k) {
                        sum += inp[i * embed_dim + k] * w[k * embed_dim + j];
                    }
                    V.data[i * embed_dim + j] = sum;
                }
            }
        }
        
        Tensor QS = Q.view({batch, seq_len, num_heads, head_dim});
        Tensor KS = K.view({batch, seq_len, num_heads, head_dim});
        Tensor VS = V.view({batch, seq_len, num_heads, head_dim});
        
        Tensor scores({batch, num_heads, seq_len, seq_len}, false);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t j = 0; j < seq_len; ++j) {
                        float sum = 0.0f;
                        for (size_t d = 0; d < head_dim; ++d) {
                            size_t qi = b * seq_len * num_heads * head_dim + i * num_heads * head_dim + h * head_dim + d;
                            size_t ki = b * seq_len * num_heads * head_dim + j * num_heads * head_dim + h * head_dim + d;
                            sum += QS.data[qi] * KS.data[ki];
                        }
                        sum *= scale;
                        if (j > i) sum = -1e9f;
                        size_t si = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        scores.data[si] = sum;
                    }
                }
            }
        }
        
        Tensor weights({batch, num_heads, seq_len, seq_len}, false);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t i = 0; i < seq_len; ++i) {
                    float max_v = -1e9f;
                    for (size_t j = 0; j < seq_len; ++j) {
                        size_t sidx = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        max_v = std::max(max_v, scores.data[sidx]);
                    }
                    float sum_e = 0.0f;
                    for (size_t j = 0; j < seq_len; ++j) {
                        size_t sidx = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        sum_e += std::exp(scores.data[sidx] - max_v);
                    }
                    for (size_t j = 0; j < seq_len; ++j) {
                        size_t sidx = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        weights.data[sidx] = std::exp(scores.data[sidx] - max_v) / (sum_e + EPSILON);
                    }
                }
            }
        }
        
        Tensor dV({batch, num_heads, seq_len, head_dim}, false);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        float sum = 0.0f;
                        for (size_t j = 0; j < seq_len; ++j) {
                            size_t widx = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                            size_t oidx = b * seq_len * embed_dim + i * embed_dim + h * head_dim + d;
                            sum += weights.data[widx] * g_out[oidx];
                        }
                        dV.data[b * num_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim + d] = sum;
                    }
                }
            }
        }
        
        Tensor dW({batch, num_heads, seq_len, seq_len}, false);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t j = 0; j < seq_len; ++j) {
                        float sum = 0.0f;
                        for (size_t d = 0; d < head_dim; ++d) {
                            size_t oidx = b * seq_len * embed_dim + i * embed_dim + h * head_dim + d;
                            size_t vidx = b * seq_len * num_heads * head_dim + j * num_heads * head_dim + h * head_dim + d;
                            sum += g_out[oidx] * VS.data[vidx];
                        }
                        dW.data[b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j] = sum;
                    }
                }
            }
        }
        
        Tensor dScores({batch, num_heads, seq_len, seq_len}, false);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t i = 0; i < seq_len; ++i) {
                    float row_s = 0.0f;
                    for (size_t j = 0; j < seq_len; ++j) {
                        size_t widx = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        row_s += dW.data[widx] * weights.data[widx];
                    }
                    for (size_t j = 0; j < seq_len; ++j) {
                        size_t widx = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        dScores.data[widx] = dW.data[widx] - row_s * weights.data[widx];
                    }
                }
            }
        }
        
        Tensor dQ({batch, num_heads, seq_len, head_dim}, false);
        Tensor dK({batch, num_heads, seq_len, head_dim}, false);
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        float sq = 0.0f, sk = 0.0f;
                        for (size_t j = 0; j < seq_len; ++j) {
                            size_t sidx = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                            size_t ki = b * seq_len * num_heads * head_dim + j * num_heads * head_dim + h * head_dim + d;
                            size_t qi = b * seq_len * num_heads * head_dim + i * num_heads * head_dim + h * head_dim + d;
                            sq += dScores.data[sidx] * KS.data[ki] * scale;
                            sk += dScores.data[sidx] * QS.data[qi] * scale;
                        }
                        dQ.data[b * num_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim + d] = sq;
                        dK.data[b * num_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim + d] = sk;
                    }
                }
            }
        }
        
        if (q_weight && q_weight->grad && q_weight->grad->data && input && input->data) {
            float* gw = q_weight->grad->data.get();
            const float* inp = input->data.get() + input->offset;
            const float* dq = dQ.data.get();
            for (size_t b = 0; b < batch; ++b) {
                for (size_t s = 0; s < seq_len; ++s) {
                    for (size_t e = 0; e < embed_dim; ++e) {
                        float g = 0.0f;
                        size_t h = e / head_dim, d = e % head_dim;
                        size_t dq_idx = b * seq_len * num_heads * head_dim + s * num_heads * head_dim + h * head_dim + d;
                        g = dq[dq_idx] * inp[(b * seq_len + s) * embed_dim + e];
                        gw[e * embed_dim + e] += g;
                    }
                }
            }
        }
        
        if (input && input->grad && input->grad->data) {
            float* g_in = input->grad->data.get() + input->grad->offset;
            const float* dq = dQ.data.get();
            const float* dk = dK.data.get();
            const float* dv = dV.data.get();
            for (size_t b = 0; b < batch; ++b) {
                for (size_t s = 0; s < seq_len; ++s) {
                    for (size_t e = 0; e < embed_dim; ++e) {
                        size_t h = e / head_dim, d = e % head_dim;
                        size_t dq_idx = b * seq_len * num_heads * head_dim + s * num_heads * head_dim + h * head_dim + d;
                        size_t gidx = (b * seq_len + s) * embed_dim + e;
                        g_in[gidx] += dq[dq_idx] + dk[dq_idx] + dv[dq_idx];
                    }
                }
            }
        }
    }
    
    std::vector<Tensor*> get_inputs() const override {
        return {input, q_weight, k_weight, v_weight, out_weight};
    }

    std::unique_ptr<Operation> clone() const override {
        return std::make_unique<AttentionBackwardOp>(input, q_weight, k_weight, v_weight, out_weight,
                                                      batch, seq_len, embed_dim, num_heads, head_dim);
    }
};

struct Attention {
    size_t embed_dim;
    size_t num_heads;
    size_t head_dim;
    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;
    bool use_rope;
    
    Attention(size_t embed_dim, size_t num_heads, bool use_rope = false)
        : embed_dim(embed_dim), num_heads(num_heads), head_dim(embed_dim / num_heads),
          q_proj(embed_dim, embed_dim, false),
          k_proj(embed_dim, embed_dim, false),
          v_proj(embed_dim, embed_dim, false),
          out_proj(embed_dim, embed_dim, false),
          use_rope(use_rope) {
        if (embed_dim % num_heads != 0) {
            throw std::runtime_error("embed_dim must be divisible by num_heads");
        }
    }
    
    void apply_rope(Tensor& q, Tensor& k, size_t seq_len, size_t batch, size_t heads) {
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < heads; ++h) {
                for (size_t s = 0; s < seq_len; ++s) {
                    float theta = static_cast<float>(s);
                    for (size_t d = 0; d < head_dim / 2; ++d) {
                        float freq = std::pow(10000.0f, -2.0f * d / static_cast<float>(head_dim));
                        float angle = theta * freq;
                        float cos_a = std::cos(angle);
                        float sin_a = std::sin(angle);
                        
                        size_t idx = b * heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim + d;
                        float q0 = q.data[q.offset + idx];
                        float q1 = q.data[q.offset + idx + head_dim / 2];
                        q.data[q.offset + idx] = q0 * cos_a - q1 * sin_a;
                        q.data[q.offset + idx + head_dim / 2] = q0 * sin_a + q1 * cos_a;
                        
                        float k0 = k.data[k.offset + idx];
                        float k1 = k.data[k.offset + idx + head_dim / 2];
                        k.data[k.offset + idx] = k0 * cos_a - k1 * sin_a;
                        k.data[k.offset + idx + head_dim / 2] = k0 * sin_a + k1 * cos_a;
                    }
                }
            }
        }
    }
    
    Tensor forward(const Tensor& x) {
        size_t batch = 1, seq_len = 1;
        
        Tensor input = x;
        if (x.shape.n == 3) {
            batch = x.shape[0];
            seq_len = x.shape[1];
            input = x.view({batch * seq_len, embed_dim});
        } else if (x.shape.n == 2) {
            batch = x.shape[0];
            seq_len = 1;
            input = x.view({batch * seq_len, embed_dim});
        } else {
            input = x.view({1, embed_dim});
        }
        
        Tensor q = q_proj.forward(input);
        Tensor k = k_proj.forward(input);
        Tensor v = v_proj.forward(input);
        
        q = q.view({batch, seq_len, num_heads, head_dim});
        k = k.view({batch, seq_len, num_heads, head_dim});
        v = v.view({batch, seq_len, num_heads, head_dim});
        
        if (use_rope) apply_rope(q, k, seq_len, batch, num_heads);
        
        Tensor scores({batch, num_heads, seq_len, seq_len}, false);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t j = 0; j < seq_len; ++j) {
                        float sum = 0.0f;
                        for (size_t d = 0; d < head_dim; ++d) {
                            size_t q_idx = b * seq_len * num_heads * head_dim + i * num_heads * head_dim + h * head_dim + d;
                            size_t k_idx = b * seq_len * num_heads * head_dim + j * num_heads * head_dim + h * head_dim + d;
                            sum += q.data[q.offset + q_idx] * k.data[k.offset + k_idx];
                        }
                        sum *= scale;
                        if (j > i) sum = -1e9f;
                        size_t s_idx = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        scores.data[scores.offset + s_idx] = sum;
                    }
                }
            }
        }
        
        Tensor weights({batch, num_heads, seq_len, seq_len}, false);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t i = 0; i < seq_len; ++i) {
                    float max_val = -1e9f;
                    for (size_t j = 0; j < seq_len; ++j) {
                        size_t s_idx = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        max_val = std::max(max_val, scores.data[scores.offset + s_idx]);
                    }
                    float sum_exp = 0.0f;
                    for (size_t j = 0; j < seq_len; ++j) {
                        size_t s_idx = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        sum_exp += std::exp(scores.data[scores.offset + s_idx] - max_val);
                    }
                    for (size_t j = 0; j < seq_len; ++j) {
                        size_t s_idx = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        weights.data[weights.offset + s_idx] = std::exp(scores.data[scores.offset + s_idx] - max_val) / (sum_exp + EPSILON);
                    }
                }
            }
        }
        
        Tensor attn_out({batch, num_heads, seq_len, head_dim}, false);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t d = 0; d < head_dim; ++d) {
                        float sum = 0.0f;
                        for (size_t j = 0; j < seq_len; ++j) {
                            size_t w_idx = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                            size_t v_idx = b * num_heads * seq_len * head_dim + j * num_heads * head_dim + h * head_dim + d;
                            sum += weights.data[weights.offset + w_idx] * v.data[v.offset + v_idx];
                        }
                        attn_out.data[attn_out.offset + b * num_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim + d] = sum;
                    }
                }
            }
        }
        
        Tensor merged({batch, seq_len, embed_dim}, false);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                for (size_t e = 0; e < embed_dim; ++e) {
                    size_t h = e / head_dim, d = e % head_dim;
                    size_t in_idx = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim + d;
                    merged.data[merged.offset + b * seq_len * embed_dim + s * embed_dim + e] = attn_out.data[attn_out.offset + in_idx];
                }
            }
        }
        
        Tensor output = out_proj.forward(merged.view({batch * seq_len, embed_dim}));
        output = output.view({batch, seq_len, embed_dim});
        
        if (x.requires_grad) {
            output.set_requires_grad(true);
            output.op = std::make_unique<AttentionBackwardOp>(
                const_cast<Tensor*>(&x),
                const_cast<Tensor*>(&q_proj.weight),
                const_cast<Tensor*>(&k_proj.weight),
                const_cast<Tensor*>(&v_proj.weight),
                const_cast<Tensor*>(&out_proj.weight),
                batch, seq_len, embed_dim, num_heads, head_dim);
            output.inputs.clear();
            output.inputs.push_back(const_cast<Tensor*>(&x));
            output.inputs.push_back(&q_proj.weight);
            output.inputs.push_back(&k_proj.weight);
            output.inputs.push_back(&v_proj.weight);
            output.inputs.push_back(&out_proj.weight);
        }
        
        return output;
    }
};

}
}