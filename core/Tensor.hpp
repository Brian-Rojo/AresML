#pragma once

#include <array>
#include <cstring>
#include <cmath>
#include <string>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <unordered_set>
#include <unordered_map>

namespace aresml {

constexpr size_t MAX_DIMS = 8;
constexpr float EPSILON = 1e-8f;

struct Shape {
    std::array<size_t, MAX_DIMS> d{};
    size_t n = 0;

    Shape() = default;
    Shape(std::initializer_list<size_t> init) {
        for (auto it = init.begin(); it != init.end() && n < MAX_DIMS; ++it) {
            d[n++] = *it;
        }
    }

    size_t size() const {
        size_t s = 1;
        for (size_t i = 0; i < n; ++i) s *= d[i];
        return s;
    }

    size_t operator[](size_t i) const { return d[i]; }
    size_t& operator[](size_t i) { return d[i]; }

    bool operator==(const Shape& o) const {
        if (n != o.n) return false;
        for (size_t i = 0; i < n; ++i) if (d[i] != o.d[i]) return false;
        return true;
    }
    bool operator!=(const Shape& o) const { return !(*this == o); }
};

// Forward declarations
struct Tensor;

// ============================================================================
// Operation - base class for all backward operations
// ============================================================================
struct Operation {
    virtual ~Operation() = default;
    virtual void backward(Tensor& grad_output) = 0;
    virtual std::vector<Tensor*> get_inputs() const = 0;
    virtual std::unique_ptr<Operation> clone() const = 0;
};

// ============================================================================
// Tensor - computation graph node
// ============================================================================
struct Tensor {
    std::shared_ptr<float[]> data;
    Shape shape;
    std::array<size_t, MAX_DIMS> strides{};
    size_t offset = 0;
    bool requires_grad = false;
    bool is_leaf = false;

    // Graph linkage
    std::vector<Tensor*> inputs;          // Input tensors that produced this tensor
    std::shared_ptr<Tensor> grad;         // Gradient accumulator
    std::unique_ptr<Operation> op;        // Operation that created this tensor

    // Metadata
    uint64_t generation = 0;
    bool is_view = false;
    Tensor* base = nullptr;
    Tensor* owner = nullptr;              // Points to the Tensor that owns this (for parameters)

    Tensor() = default;

    explicit Tensor(const Shape& s, bool rg = false) : shape(s), requires_grad(rg) {
        if (shape.size() > 0) {
            data = std::shared_ptr<float[]>(new float[shape.size()]());
        }
        compute_contiguous_strides();
        if (requires_grad) {
            grad = std::make_shared<Tensor>(shape);
            grad->zero_();
            
            // Create node for graph tracking
            // This tensor is a leaf by default (root of computation)
        }
    }

    Tensor(const Tensor& other) : data(other.data), shape(other.shape), strides(other.strides),
                                   offset(other.offset), requires_grad(other.requires_grad),
                                   is_leaf(other.is_leaf), generation(other.generation),
                                   is_view(other.is_view), base(other.base), owner(other.owner) {
        if (other.grad) {
            grad = std::make_shared<Tensor>(*other.grad);
        }
        // NOTE: Do NOT clone op - it will be recreated by the layer
        inputs = other.inputs;
    }

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            data = other.data;
            shape = other.shape;
            strides = other.strides;
            offset = other.offset;
            requires_grad = other.requires_grad;
            is_leaf = other.is_leaf;
            generation = other.generation;
            is_view = other.is_view;
            base = other.base;
            owner = other.owner;
            if (other.grad) grad = std::make_shared<Tensor>(*other.grad);
            inputs = other.inputs;
            // NOTE: Do NOT clone op
            op.reset();
        }
        return *this;
    }

    void compute_contiguous_strides() {
        strides.fill(1);
        if (shape.n == 0) return;
        for (size_t i = shape.n - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * shape[i];
        }
    }

    void zero_() {
        if (!data) return;
        size_t n = shape.size();
        for (size_t i = 0; i < n; ++i) {
            data[offset + i] = 0.0f;
        }
    }

    void fill(float v) {
        if (!data) return;
        size_t n = shape.size();
        for (size_t i = 0; i < n; ++i) {
            data[offset + i] = v;
        }
    }

    void set_requires_grad(bool rg) {
        requires_grad = rg;
        if (rg && !grad) {
            grad = std::make_shared<Tensor>(shape);
            grad->zero_();
        }
        if (!rg) {
            grad.reset();
        }
    }

    bool has_nan() const {
        if (!data) return false;
        for (size_t i = 0; i < shape.size(); ++i) {
            float v = data[offset + i];
            if (std::isnan(v) || std::isinf(v)) return true;
        }
        return false;
    }

    Tensor view(const Shape& new_shape) const {
        Tensor result;
        result.data = data;
        result.shape = new_shape;
        result.strides = strides;
        result.offset = offset;
        result.requires_grad = requires_grad;
        result.is_view = true;
        result.base = const_cast<Tensor*>(this);
        result.owner = owner;
        result.compute_contiguous_strides();

        if (requires_grad) {
            result.grad = std::make_shared<Tensor>(new_shape);
            result.grad->zero_();
            result.inputs.clear();
            result.inputs.push_back(const_cast<Tensor*>(this));
        }

        return result;
    }

    Tensor reshape(const Shape& new_shape) const {
        return view(new_shape);
    }

    Tensor transpose(int dim0, int dim1) const {
        Tensor result;
        result.data = data;
        result.shape = shape;
        result.strides = strides;
        result.offset = offset;
        result.requires_grad = requires_grad;
        result.is_view = true;
        result.base = const_cast<Tensor*>(this);
        result.owner = owner;
        
        std::swap(result.shape.d[dim0], result.shape.d[dim1]);
        result.strides[dim0] = strides[dim1];
        result.strides[dim1] = strides[dim0];

        if (requires_grad) {
            result.grad = std::make_shared<Tensor>(result.shape);
            result.grad->zero_();
            result.inputs.clear();
            result.inputs.push_back(const_cast<Tensor*>(this));
        }

        return result;
    }

    Tensor permute(const std::vector<int>& dims) const {
        Tensor result;
        result.data = data;
        result.requires_grad = requires_grad;
        result.is_view = true;
        result.base = const_cast<Tensor*>(this);
        result.owner = owner;
        
        result.shape.n = dims.size();
        for (size_t i = 0; i < dims.size(); ++i) {
            result.shape.d[i] = shape.d[dims[i]];
            result.strides[i] = strides[dims[i]];
        }

        if (requires_grad) {
            result.grad = std::make_shared<Tensor>(result.shape);
            result.grad->zero_();
            result.inputs.clear();
            result.inputs.push_back(const_cast<Tensor*>(this));
        }

        return result;
    }

    bool is_contiguous() const {
        size_t expected_stride = 1;
        for (int i = shape.n - 1; i >= 0; --i) {
            if (strides[i] != expected_stride) return false;
            expected_stride *= shape.d[i];
        }
        return true;
    }

    Tensor contiguous() const {
        if (is_contiguous()) return *this;
        
        Tensor result(shape, requires_grad);
        if (data && result.data) {
            for (size_t i = 0; i < shape.size(); ++i) {
                result.data[i] = data[offset + i];
            }
        }
        result.is_leaf = is_leaf;
        result.owner = owner;
        
        if (requires_grad) {
            result.inputs = inputs;
            result.grad = std::make_shared<Tensor>(shape);
            result.grad->zero_();
        }
        
        return result;
    }

    float sum() const {
        if (!data) return 0.0f;
        float s = 0.0f;
        for (size_t i = 0; i < shape.size(); ++i) {
            s += data[offset + i];
        }
        return s;
    }

    Tensor clone() const {
        Tensor result(shape, requires_grad);
        if (data && result.data) {
            std::memcpy(result.data.get(), data.get(), shape.size() * sizeof(float));
        }
        result.is_leaf = is_leaf;
        result.owner = owner;
        result.inputs = inputs;

        if (requires_grad) {
            result.grad = std::make_shared<Tensor>(shape);
            result.grad->zero_();
        }

        return result;
    }

    // Iterator support for range-based for loops
    float* begin() { return data.get() + offset; }
    float* end() { return data.get() + offset + shape.size(); }
    const float* begin() const { return data.get() + offset; }
    const float* end() const { return data.get() + offset + shape.size(); }
};

// ============================================================================
// ViewReshapeBackwardOp - BACKWARD for view/reshape operations
// Propagates gradient back to the source tensor by reshape
// ============================================================================
struct ViewReshapeBackwardOp : Operation {
    std::vector<Tensor*> source_tensors;
    std::vector<size_t> source_sizes;

    ViewReshapeBackwardOp() = default;

    ViewReshapeBackwardOp(Tensor* src, size_t src_size) {
        source_tensors.push_back(src);
        source_sizes.push_back(src_size);
    }

    void backward(Tensor& grad) override {
        if (!grad.grad || !grad.grad->data) return;
        
        const float* g_grad = grad.grad->data.get() + grad.grad->offset;
        size_t n = grad.shape.size();

        for (size_t j = 0; j < source_tensors.size(); ++j) {
            auto* src = source_tensors[j];
            if (!src || !src->grad || !src->grad->data) continue;
            
            float* g_src = src->grad->data.get() + src->grad->offset;
            size_t src_n = source_sizes[j];
            size_t min_n = (n < src_n) ? n : src_n;
            
            for (size_t i = 0; i < min_n; ++i) {
                g_src[i] += g_grad[i];
            }
        }
    }

    std::vector<Tensor*> get_inputs() const override {
        return source_tensors;
    }

    std::unique_ptr<Operation> clone() const override {
        auto op = std::make_unique<ViewReshapeBackwardOp>();
        op->source_tensors = source_tensors;
        op->source_sizes = source_sizes;
        return op;
    }
};

// ============================================================================
// Helper functions
// ============================================================================

inline Tensor tensor_zeros(const Shape& s, bool rg = false) {
    Tensor t(s, rg);
    t.zero_();
    return t;
}

inline Tensor tensor_randn(const Shape& s, bool rg = false) {
    Tensor t(s, rg);
    if (t.data) {
        for (size_t i = 0; i < t.shape.size(); ++i) {
            float u1 = static_cast<float>(rand()) / RAND_MAX;
            float u2 = static_cast<float>(rand()) / RAND_MAX;
            u1 = std::max(u1, 1e-10f); // Avoid log(0)
            t.data[i] = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159f * u2);
        }
    }
    return t;
}

inline Tensor tensor_ones(const Shape& s, bool rg = false) {
    Tensor t(s, rg);
    t.fill(1.0f);
    return t;
}

} // namespace aresml
