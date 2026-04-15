#pragma once

#include "../core/Tensor.hpp"
#include "../core/Autograd.hpp"
#include "../core/simd/Simd.hpp"
#include "../core/threading/Parallel.hpp"
#include <cmath>
#include <vector>

namespace aresml {
namespace ops {

inline float sigmoid_float(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float silu_float(float x) {
    return x / (1.0f + std::exp(-x));
}

inline float relu_float(float x) {
    return x > 0.0f ? x : 0.0f;
}

inline float gelu_float(float x) {
    float sqrt_2_over_pi = std::sqrt(2.0f / 3.14159265f);
    return 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

// ============================================================================
// AddOp
// ============================================================================
struct AddOp : Operation {
    Tensor* A;
    Tensor* B;

    AddOp(Tensor* a, Tensor* b) : A(a), B(b) {}

    void backward(Tensor& grad) override {
        if (!grad.grad || !grad.grad->data) return;
        float* gout = grad.grad->data.get() + grad.grad->offset;
        size_t n = grad.shape.size();

        if (A && A->grad && A->grad->data) {
            float* gA = A->grad->data.get() + A->grad->offset;
            for (size_t i = 0; i < n; ++i) {
                gA[i] += gout[i];
            }
        }

        if (B && B->grad && B->grad->data) {
            float* gB = B->grad->data.get() + B->grad->offset;
            for (size_t i = 0; i < n; ++i) {
                gB[i] += gout[i];
            }
        }
    }

    std::vector<Tensor*> get_inputs() const override {
        return {A, B};
    }

    std::unique_ptr<Operation> clone() const override {
        return std::make_unique<AddOp>(A, B);
    }
};

// ============================================================================
// MulOp - Element-wise multiplication
// ============================================================================
struct MulOp : Operation {
    Tensor* A;
    Tensor* B;

    MulOp(Tensor* a, Tensor* b) : A(a), B(b) {}

    void backward(Tensor& grad) override {
        if (!grad.grad || !grad.grad->data) return;
        float* gout = grad.grad->data.get() + grad.grad->offset;
        size_t n = grad.shape.size();

        if (A && A->grad && A->grad->data && B && B->data) {
            float* gA = A->grad->data.get() + A->grad->offset;
            const float* b_data = B->data.get() + B->offset;
            for (size_t i = 0; i < n; ++i) {
                gA[i] += gout[i] * b_data[i];
            }
        }

        if (B && B->grad && B->grad->data && A && A->data) {
            float* gB = B->grad->data.get() + B->grad->offset;
            const float* a_data = A->data.get() + A->offset;
            for (size_t i = 0; i < n; ++i) {
                gB[i] += gout[i] * a_data[i];
            }
        }
    }

    std::vector<Tensor*> get_inputs() const override {
        return {A, B};
    }

    std::unique_ptr<Operation> clone() const override {
        return std::make_unique<MulOp>(A, B);
    }
};

// ============================================================================
// MatmulOp - Matrix multiplication
// ============================================================================
struct MatmulOp : Operation {
    Tensor* A;
    Tensor* B;

    MatmulOp(Tensor* a, Tensor* b) : A(a), B(b) {}

    void backward(Tensor& grad) override {
        if (!grad.grad || !grad.grad->data) return;
        
        size_t M = grad.shape[0];
        size_t N = grad.shape[1];
        size_t K = A ? A->shape[1] : 0;
        
        const float* gout = grad.grad->data.get() + grad.grad->offset;
        
        if (A && A->grad && A->grad->data && B && B->data) {
            float* gA = A->grad->data.get() + A->grad->offset;
            const float* b_data = B->data.get() + B->offset;
            
            for (size_t i = 0; i < M; ++i) {
                for (size_t k = 0; k < K; ++k) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < N; ++j) {
                        sum += gout[i * N + j] * b_data[k * N + j];
                    }
                    gA[i * K + k] += sum;
                }
            }
        }
        
        if (B && B->grad && B->grad->data && A && A->data) {
            float* gB = B->grad->data.get() + B->grad->offset;
            const float* a_data = A->data.get() + A->offset;
            
            for (size_t k = 0; k < K; ++k) {
                for (size_t j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (size_t i = 0; i < M; ++i) {
                        sum += gout[i * N + j] * a_data[i * K + k];
                    }
                    gB[k * N + j] += sum;
                }
            }
        }
    }

    std::vector<Tensor*> get_inputs() const override {
        return {A, B};
    }

    std::unique_ptr<Operation> clone() const override {
        return std::make_unique<MatmulOp>(A, B);
    }
};

// ============================================================================
// ReluOp
// ============================================================================
struct ReluOp : Operation {
    Tensor* input;

    ReluOp(Tensor* inp) : input(inp) {}

    void backward(Tensor& grad) override {
        if (!input->grad || !input->grad->data || !grad.grad) return;

        const float* in_data = input->data.get() + input->offset;
        float* g_in = input->grad->data.get() + input->grad->offset;
        const float* g_out = grad.grad->data.get() + grad.grad->offset;
        size_t n = input->shape.size();

        for (size_t i = 0; i < n; ++i) {
            if (in_data[i] > 0.0f) {
                g_in[i] += g_out[i];
            }
        }
    }

    std::vector<Tensor*> get_inputs() const override {
        return {input};
    }

    std::unique_ptr<Operation> clone() const override {
        return std::make_unique<ReluOp>(input);
    }
};

// ============================================================================
// SiluOp
// ============================================================================
struct SiluOp : Operation {
    Tensor* input;

    SiluOp(Tensor* inp) : input(inp) {}

    void backward(Tensor& grad) override {
        if (!input->grad || !input->grad->data || !grad.grad) return;

        const float* in_data = input->data.get() + input->offset;
        float* g_in = input->grad->data.get() + input->grad->offset;
        const float* g_out = grad.grad->data.get() + grad.grad->offset;
        size_t n = input->shape.size();

        for (size_t i = 0; i < n; ++i) {
            float x = in_data[i];
            float sig = sigmoid_float(x);
            float d_si = sig * (1.0f + x * (1.0f - sig));
            g_in[i] += g_out[i] * d_si;
        }
    }

    std::vector<Tensor*> get_inputs() const override {
        return {input};
    }

    std::unique_ptr<Operation> clone() const override {
        return std::make_unique<SiluOp>(input);
    }
};

// ============================================================================
// GeluOp
// ============================================================================
struct GeluOp : Operation {
    Tensor* input;

    GeluOp(Tensor* inp) : input(inp) {}

    void backward(Tensor& grad) override {
        if (!input->grad || !input->grad->data || !grad.grad) return;

        const float* in_data = input->data.get() + input->offset;
        float* g_in = input->grad->data.get() + input->grad->offset;
        const float* g_out = grad.grad->data.get() + grad.grad->offset;
        size_t n = input->shape.size();

        float sqrt_2_over_pi = std::sqrt(2.0f / 3.14159265f);

        for (size_t i = 0; i < n; ++i) {
            float x = in_data[i];
            float t = std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x));
            float d_gelu = 0.5f * t + 0.5f * x * (1.0f - t * t) * (sqrt_2_over_pi * (1.0f + 0.134f * x) + 0.044715f * 3.0f * x * x);
            g_in[i] += g_out[i] * d_gelu;
        }
    }

    std::vector<Tensor*> get_inputs() const override {
        return {input};
    }

    std::unique_ptr<Operation> clone() const override {
        return std::make_unique<GeluOp>(input);
    }
};

// ============================================================================
// SoftmaxOp
// ============================================================================
struct SoftmaxOp : Operation {
    Tensor* input;
    int axis;

    SoftmaxOp(Tensor* inp, int axis) : input(inp), axis(axis) {}

    void backward(Tensor& grad) override {
        if (!grad.grad || !grad.grad->data) return;
        if (!input || !input->grad || !input->grad->data) return;

        const float* out_data = grad.data.get() + grad.offset;
        float* g_in = input->grad->data.get() + input->grad->offset;
        const float* g_out = grad.grad->data.get() + grad.grad->offset;

        size_t total_size = grad.shape.size();
        if (total_size == 0) return;

        if (grad.shape.n == 2) {
            size_t rows = grad.shape[0];
            size_t cols = grad.shape[1];

            for (size_t r = 0; r < rows; ++r) {
                float sum_soft_g = 0.0f;
                for (size_t c = 0; c < cols; ++c) {
                    size_t idx = r * cols + c;
                    sum_soft_g += out_data[idx] * g_out[idx];
                }

                for (size_t c = 0; c < cols; ++c) {
                    size_t idx = r * cols + c;
                    g_in[idx] += out_data[idx] * (g_out[idx] - sum_soft_g);
                }
            }
        }
        else if (grad.shape.n == 3) {
            size_t batch = grad.shape[0];
            size_t seq = grad.shape[1];
            size_t vocab = grad.shape[2];

            for (size_t b = 0; b < batch; ++b) {
                for (size_t s = 0; s < seq; ++s) {
                    float sum_soft_g = 0.0f;
                    for (size_t v = 0; v < vocab; ++v) {
                        size_t idx = b * seq * vocab + s * vocab + v;
                        sum_soft_g += out_data[idx] * g_out[idx];
                    }

                    for (size_t v = 0; v < vocab; ++v) {
                        size_t idx = b * seq * vocab + s * vocab + v;
                        g_in[idx] += out_data[idx] * (g_out[idx] - sum_soft_g);
                    }
                }
            }
        }
        else {
            float sum_soft_g = 0.0f;
            for (size_t i = 0; i < total_size; ++i) {
                sum_soft_g += out_data[i] * g_out[i];
            }

            for (size_t i = 0; i < total_size; ++i) {
                g_in[i] += out_data[i] * (g_out[i] - sum_soft_g);
            }
        }
    }

    std::vector<Tensor*> get_inputs() const override {
        return {input};
    }

    std::unique_ptr<Operation> clone() const override {
        return std::make_unique<SoftmaxOp>(input, axis);
    }
};

// ============================================================================
// LogSoftmaxOp
// ============================================================================
struct LogSoftmaxOp : Operation {
    Tensor* input;
    int axis;

    LogSoftmaxOp(Tensor* inp, int axis) : input(inp), axis(axis) {}

    void backward(Tensor& grad) override {
        if (!grad.grad || !grad.grad->data) return;
        if (!input || !input->grad || !input->grad->data) return;

        const float* out_data = grad.data.get() + grad.offset;
        float* g_in = input->grad->data.get() + input->grad->offset;
        const float* g_out = grad.grad->data.get() + grad.grad->offset;

        size_t total_size = grad.shape.size();
        if (total_size == 0) return;

        if (grad.shape.n == 2) {
            size_t rows = grad.shape[0];
            size_t cols = grad.shape[1];

            for (size_t r = 0; r < rows; ++r) {
                float sum_g_out = 0.0f;
                for (size_t c = 0; c < cols; ++c) {
                    size_t idx = r * cols + c;
                    sum_g_out += g_out[idx];
                }

                for (size_t c = 0; c < cols; ++c) {
                    size_t idx = r * cols + c;
                    g_in[idx] += g_out[idx] - std::exp(out_data[idx]) * sum_g_out;
                }
            }
        }
        else if (grad.shape.n == 3) {
            size_t batch = grad.shape[0];
            size_t seq = grad.shape[1];
            size_t vocab = grad.shape[2];

            for (size_t b = 0; b < batch; ++b) {
                for (size_t s = 0; s < seq; ++s) {
                    float sum_g_out = 0.0f;
                    for (size_t v = 0; v < vocab; ++v) {
                        size_t idx = b * seq * vocab + s * vocab + v;
                        sum_g_out += g_out[idx];
                    }

                    for (size_t v = 0; v < vocab; ++v) {
                        size_t idx = b * seq * vocab + s * vocab + v;
                        g_in[idx] += g_out[idx] - std::exp(out_data[idx]) * sum_g_out;
                    }
                }
            }
        }
        else {
            float sum_g_out = 0.0f;
            for (size_t i = 0; i < total_size; ++i) {
                sum_g_out += g_out[i];
            }

            for (size_t i = 0; i < total_size; ++i) {
                g_in[i] += g_out[i] - std::exp(out_data[i]) * sum_g_out;
            }
        }
    }

    std::vector<Tensor*> get_inputs() const override {
        return {input};
    }

    std::unique_ptr<Operation> clone() const override {
        return std::make_unique<LogSoftmaxOp>(input, axis);
    }
};

// ============================================================================
// SumOp
// ============================================================================
struct SumOp : Operation {
    Tensor* input;

    SumOp(Tensor* inp) : input(inp) {}

    std::vector<Tensor*> get_inputs() const override {
        return {input};
    }

    void backward(Tensor& grad) override {
        if (!grad.grad || !grad.grad->data) return;
        if (!input || !input->grad) return;

        float g = grad.grad->data[grad.grad->offset];
        float* gin = input->grad->data.get() + input->grad->offset;
        size_t n = input->shape.size();

        for (size_t i = 0; i < n; ++i) {
            gin[i] += g;
        }
    }

    std::unique_ptr<Operation> clone() const override {
        return std::make_unique<SumOp>(input);
    }
};

// ============================================================================
// Element-wise functions
// ============================================================================

inline Tensor relu(const Tensor& x) {
    Tensor out(x.shape, x.requires_grad);

    const float* in = x.data.get() + x.offset;
    float* o = out.data.get() + out.offset;
    size_t n = x.shape.size();

    for (size_t i = 0; i < n; ++i) {
        o[i] = relu_float(in[i]);
    }

    if (x.requires_grad) {
        out.op = std::make_unique<ReluOp>(const_cast<Tensor*>(&x));
        out.inputs.clear();
        out.inputs.push_back(const_cast<Tensor*>(&x));
    }

    return out;
}

inline Tensor silu(const Tensor& x) {
    Tensor out(x.shape, x.requires_grad);

    const float* in = x.data.get() + x.offset;
    float* o = out.data.get() + out.offset;
    size_t n = x.shape.size();

    for (size_t i = 0; i < n; ++i) {
        o[i] = silu_float(in[i]);
    }

    if (x.requires_grad) {
        out.op = std::make_unique<SiluOp>(const_cast<Tensor*>(&x));
        out.inputs.clear();
        out.inputs.push_back(const_cast<Tensor*>(&x));
    }

    return out;
}

inline Tensor gelu(const Tensor& x) {
    Tensor out(x.shape, x.requires_grad);

    const float* in = x.data.get() + x.offset;
    float* o = out.data.get() + out.offset;
    size_t n = x.shape.size();

    for (size_t i = 0; i < n; ++i) {
        o[i] = gelu_float(in[i]);
    }

    if (x.requires_grad) {
        out.op = std::make_unique<GeluOp>(const_cast<Tensor*>(&x));
        out.inputs.clear();
        out.inputs.push_back(const_cast<Tensor*>(&x));
    }

    return out;
}

inline Tensor add(const Tensor& a, const Tensor& b) {
    Shape out_shape = a.shape;

    Tensor out(out_shape, a.requires_grad || b.requires_grad);

    const float* ad = a.data.get() + a.offset;
    const float* bd = b.data.get() + b.offset;
    float* od = out.data.get() + out.offset;
    size_t n = out.shape.size();

    for (size_t i = 0; i < n; ++i) {
        od[i] = ad[i] + bd[i];
    }

    if (a.requires_grad || b.requires_grad) {
        out.op = std::make_unique<AddOp>(const_cast<Tensor*>(&a), const_cast<Tensor*>(&b));
        out.inputs.clear();
        if (a.requires_grad) out.inputs.push_back(const_cast<Tensor*>(&a));
        if (b.requires_grad) out.inputs.push_back(const_cast<Tensor*>(&b));
    }

    return out;
}

inline Tensor mul(const Tensor& a, const Tensor& b) {
    Shape out_shape = a.shape;

    Tensor out(out_shape, a.requires_grad || b.requires_grad);

    const float* ad = a.data.get() + a.offset;
    const float* bd = b.data.get() + b.offset;
    float* od = out.data.get() + out.offset;
    size_t n = out.shape.size();

    for (size_t i = 0; i < n; ++i) {
        od[i] = ad[i] * bd[i];
    }

    if (a.requires_grad || b.requires_grad) {
        out.op = std::make_unique<MulOp>(const_cast<Tensor*>(&a), const_cast<Tensor*>(&b));
        out.inputs.clear();
        if (a.requires_grad) out.inputs.push_back(const_cast<Tensor*>(&a));
        if (b.requires_grad) out.inputs.push_back(const_cast<Tensor*>(&b));
    }

    return out;
}

inline Tensor matmul(const Tensor& a, const Tensor& b) {
    Shape out_shape = {a.shape[0], b.shape[1]};
    Tensor out(out_shape, a.requires_grad || b.requires_grad);

    const float* ad = a.data.get() + a.offset;
    const float* bd = b.data.get() + b.offset;
    float* od = out.data.get() + out.offset;
    
    size_t M = a.shape[0];
    size_t K = a.shape[1];
    size_t N = b.shape[1];

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += ad[i * K + k] * bd[k * N + j];
            }
            od[i * N + j] = sum;
        }
    }

    if (a.requires_grad || b.requires_grad) {
        out.op = std::make_unique<MatmulOp>(const_cast<Tensor*>(&a), const_cast<Tensor*>(&b));
        out.inputs.clear();
        if (a.requires_grad) out.inputs.push_back(const_cast<Tensor*>(&a));
        if (b.requires_grad) out.inputs.push_back(const_cast<Tensor*>(&b));
    }

    return out;
}

inline Tensor sum(const Tensor& a) {
    Tensor out({1}, a.requires_grad);

    const float* ad = a.data.get() + a.offset;
    float result = 0.0f;
    for (size_t i = 0; i < a.shape.size(); ++i) {
        result += ad[i];
    }

    out.data[0] = result;

    if (a.requires_grad) {
        out.op = std::make_unique<SumOp>(const_cast<Tensor*>(&a));
        out.inputs.clear();
        out.inputs.push_back(const_cast<Tensor*>(&a));
    }

    return out;
}

} // namespace ops
} // namespace aresml
