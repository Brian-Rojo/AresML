// AresML FFI Implementation - C++ to C bridge

#include "aresml_ffi.h"
#include "../core/Tensor.hpp"
#include "../core/Autograd.hpp"
#include "../nn/Linear.hpp"
#include "../loss/MSELoss.hpp"
#include "../loss/CrossEntropy.hpp"

#include <cstring>
#include <cstdlib>
#include <string>

thread_local std::string last_error_message;

extern "C" {

const char* aresml_get_last_error() {
    if (last_error_message.empty()) return nullptr;
    return last_error_message.c_str();
}

void aresml_clear_last_error() {
    last_error_message.clear();
}

// ============================================================================
// CONTEXT (wrapper around global singleton for compatibility)
// ============================================================================

AresmlGraphContext aresml_graph_context_create() {
    // For now, just return non-null to indicate context exists
    // Real implementation would create per-context state
    return reinterpret_cast<AresmlGraphContext>(0x1);
}

void aresml_graph_context_destroy(AresmlGraphContext ctx) {
    // No-op for now
    (void)ctx;
}

aresml_error_t aresml_graph_register_leaf(AresmlGraphContext ctx, AresmlTensor tensor) {
    if (!tensor) return ARESML_ERR_NULLPTR;
    auto* t = static_cast<aresml::Tensor*>(tensor);
    if (t->requires_grad && t->is_leaf) {
        aresml::get_engine().register_leaf(t);
    }
    return ARESML_SUCCESS;
    (void)ctx;
}

aresml_error_t aresml_graph_zero_grad(AresmlGraphContext ctx) {
    aresml::zero_grad();
    return ARESML_SUCCESS;
    (void)ctx;
}

aresml_error_t aresml_graph_backward(AresmlGraphContext ctx, AresmlTensor loss) {
    if (!loss) return ARESML_ERR_NULLPTR;
    auto* t = static_cast<aresml::Tensor*>(loss);
    aresml::backward(*t);
    return ARESML_SUCCESS;
    (void)ctx;
}

// ============================================================================
// TENSOR
// ============================================================================

static aresml::Shape make_shape(const int64_t* shape, int32_t ndim) {
    aresml::Shape s;
    s.n = static_cast<size_t>(ndim);
    for (int i = 0; i < ndim && i < 8; i++) {
        s.d[i] = static_cast<size_t>(shape[i]);
    }
    return s;
}

AresmlTensor aresml_tensor_create(const int64_t* shape, int32_t ndim) {
    if (!shape || ndim <= 0) return nullptr;
    try {
        auto s = make_shape(shape, ndim);
        auto* tensor = new (std::nothrow) aresml::Tensor(s, true);
        return tensor;
    } catch (...) {
        return nullptr;
    }
}

AresmlTensor aresml_tensor_randn(const int64_t* shape, int32_t ndim) {
    if (!shape || ndim <= 0) return nullptr;
    auto s = make_shape(shape, ndim);
    aresml::Tensor t = aresml::tensor_randn(s, true);
    return new (std::nothrow) aresml::Tensor(std::move(t));
}

AresmlTensor aresml_tensor_zeros(const int64_t* shape, int32_t ndim) {
    if (!shape || ndim <= 0) return nullptr;
    auto s = make_shape(shape, ndim);
    aresml::Tensor t = aresml::tensor_zeros(s, true);
    return new (std::nothrow) aresml::Tensor(std::move(t));
}

AresmlTensor aresml_tensor_clone(AresmlTensor tensor) {
    if (!tensor) return nullptr;
    auto* t = static_cast<aresml::Tensor*>(tensor);
    return new (std::nothrow) aresml::Tensor(*t);
}

void aresml_tensor_free(AresmlTensor tensor) {
    if (tensor) delete static_cast<aresml::Tensor*>(tensor);
}

int64_t* aresml_tensor_get_shape(AresmlTensor tensor, int32_t* out_ndim) {
    if (!tensor || !out_ndim) return nullptr;
    auto* t = static_cast<aresml::Tensor*>(tensor);
    *out_ndim = static_cast<int32_t>(t->shape.n);
    int64_t* result = new int64_t[t->shape.n];
    for (size_t i = 0; i < t->shape.n; i++) {
        result[i] = static_cast<int64_t>(t->shape[i]);
    }
    return result;
}

int64_t aresml_tensor_size(AresmlTensor tensor) {
    if (!tensor) return 0;
    return static_cast<int64_t>(static_cast<aresml::Tensor*>(tensor)->shape.size());
}

const float* aresml_tensor_data(AresmlTensor tensor) {
    if (!tensor) return nullptr;
    auto* t = static_cast<aresml::Tensor*>(tensor);
    if (!t->data) return nullptr;
    return t->data.get() + t->offset;
}

aresml_error_t aresml_tensor_set_requires_grad(AresmlTensor tensor, bool value) {
    if (!tensor) return ARESML_ERR_NULLPTR;
    static_cast<aresml::Tensor*>(tensor)->set_requires_grad(value);
    return ARESML_SUCCESS;
}

bool aresml_tensor_requires_grad(AresmlTensor tensor) {
    if (!tensor) return false;
    return static_cast<aresml::Tensor*>(tensor)->requires_grad;
}

bool aresml_tensor_has_nan(AresmlTensor tensor) {
    if (!tensor) return false;
    return static_cast<aresml::Tensor*>(tensor)->has_nan();
}

// ============================================================================
// MATH
// ============================================================================

AresmlTensor aresml_add(AresmlTensor a, AresmlTensor b) {
    if (!a || !b) return nullptr;
    auto* ta = static_cast<aresml::Tensor*>(a);
    auto* tb = static_cast<aresml::Tensor*>(b);
    if (ta->shape != tb->shape) return nullptr;
    
    aresml::Tensor result(ta->shape, true);
    for (size_t i = 0; i < ta->shape.size(); i++) {
        result.data.get()[i] = ta->data.get()[ta->offset + i] + tb->data.get()[tb->offset + i];
    }
    return new (std::nothrow) aresml::Tensor(std::move(result));
}

AresmlTensor aresml_mul(AresmlTensor a, AresmlTensor b) {
    if (!a || !b) return nullptr;
    auto* ta = static_cast<aresml::Tensor*>(a);
    auto* tb = static_cast<aresml::Tensor*>(b);
    if (ta->shape != tb->shape) return nullptr;
    
    aresml::Tensor result(ta->shape, true);
    for (size_t i = 0; i < ta->shape.size(); i++) {
        result.data.get()[i] = ta->data.get()[ta->offset + i] * tb->data.get()[tb->offset + i];
    }
    return new (std::nothrow) aresml::Tensor(std::move(result));
}

AresmlTensor aresml_matmul(AresmlTensor a, AresmlTensor b) {
    if (!a || !b) return nullptr;
    auto* ta = static_cast<aresml::Tensor*>(a);
    auto* tb = static_cast<aresml::Tensor*>(b);
    try {
        aresml::Tensor result = aresml::backend_cpu::matmul(*ta, *tb);
        return new (std::nothrow) aresml::Tensor(std::move(result));
    } catch (...) {
        return nullptr;
    }
}

AresmlTensor aresml_relu(AresmlTensor x) {
    if (!x) return nullptr;
    auto* tx = static_cast<aresml::Tensor*>(x);
    aresml::Tensor result(tx->shape, tx->requires_grad);
    for (size_t i = 0; i < tx->shape.size(); i++) {
        float v = tx->data.get()[tx->offset + i];
        result.data.get()[i] = (v > 0) ? v : 0;
    }
    return new (std::nothrow) aresml::Tensor(std::move(result));
}

AresmlTensor aresml_softmax(AresmlTensor x, int32_t axis) {
    if (!x) return nullptr;
    auto* tx = static_cast<aresml::Tensor*>(x);
    try {
        aresml::Tensor result = aresml::backend_cpu::softmax(*tx, axis);
        return new (std::nothrow) aresml::Tensor(std::move(result));
    } catch (...) {
        return nullptr;
    }
}

AresmlTensor aresml_log_softmax(AresmlTensor x, int32_t axis) {
    if (!x) return nullptr;
    auto* tx = static_cast<aresml::Tensor*>(x);
    try {
        aresml::Tensor result = aresml::backend_cpu::log_softmax(*tx, axis);
        return new (std::nothrow) aresml::Tensor(std::move(result));
    } catch (...) {
        return nullptr;
    }
}

AresmlTensor aresml_sum(AresmlTensor x) {
    if (!x) return nullptr;
    auto* tx = static_cast<aresml::Tensor*>(x);
    float s = tx->sum();
    aresml::Tensor result(aresml::Shape({1}), tx->requires_grad);
    result.data.get()[0] = s;
    return new (std::nothrow) aresml::Tensor(std::move(result));
}

AresmlTensor aresml_mean(AresmlTensor x) {
    if (!x) return nullptr;
    auto* tx = static_cast<aresml::Tensor*>(x);
    float m = tx->mean();
    aresml::Tensor result(aresml::Shape({1}), tx->requires_grad);
    result.data.get()[0] = m;
    return new (std::nothrow) aresml::Tensor(std::move(result));
}

// ============================================================================
// LINEAR
// ============================================================================

struct LinearWrapper {
    aresml::nn::Linear layer;
    LinearWrapper(int32_t in_feat, int32_t out_feat, bool bias)
        : layer(static_cast<size_t>(in_feat), static_cast<size_t>(out_feat), bias) {}
};

AresmlLinear aresml_linear_create(int32_t in_features, int32_t out_features, bool bias) {
    try {
        return new (std::nothrow) LinearWrapper(in_features, out_features, bias);
    } catch (...) {
        return nullptr;
    }
}

AresmlTensor aresml_linear_forward(AresmlLinear layer, AresmlTensor input) {
    if (!layer || !input) return nullptr;
    auto* l = static_cast<LinearWrapper*>(layer);
    auto* inp = static_cast<aresml::Tensor*>(input);
    try {
        aresml::Tensor result = l->layer.forward(*inp);
        return new (std::nothrow) aresml::Tensor(std::move(result));
    } catch (...) {
        return nullptr;
    }
}

const float* aresml_linear_get_weight_data(AresmlLinear layer) {
    if (!layer) return nullptr;
    auto* l = static_cast<LinearWrapper*>(layer);
    if (!l->layer.weight.data) return nullptr;
    return l->layer.weight.data.get();
}

const float* aresml_linear_get_bias_data(AresmlLinear layer) {
    if (!layer) return nullptr;
    auto* l = static_cast<LinearWrapper*>(layer);
    if (!l->layer.bias.data) return nullptr;
    return l->layer.bias.data.get();
}

void aresml_linear_free(AresmlLinear layer) {
    if (layer) delete static_cast<LinearWrapper*>(layer);
}

// ============================================================================
// LOSS
// ============================================================================

AresmlTensor aresml_mse_loss(AresmlTensor pred, AresmlTensor target) {
    if (!pred || !target) return nullptr;
    auto* p = static_cast<aresml::Tensor*>(pred);
    auto* t = static_cast<aresml::Tensor*>(target);
    try {
        aresml::loss::MSELoss loss_fn;
        aresml::Tensor result = loss_fn.forward(*p, *t);
        return new (std::nothrow) aresml::Tensor(std::move(result));
    } catch (...) {
        return nullptr;
    }
}

AresmlTensor aresml_cross_entropy(AresmlTensor logits, AresmlTensor targets) {
    if (!logits || !targets) return nullptr;
    auto* l = static_cast<aresml::Tensor*>(logits);
    auto* t = static_cast<aresml::Tensor*>(targets);
    try {
        aresml::loss::CrossEntropyLoss loss_fn;
        aresml::Tensor result = loss_fn.forward(*l, *t);
        return new (std::nothrow) aresml::Tensor(std::move(result));
    } catch (...) {
        return nullptr;
    }
}

} // extern "C"