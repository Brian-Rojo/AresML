#pragma once

#include "../../core/Tensor.hpp"
#include <memory>

namespace aresml {
namespace backend {

/**
 * Abstract Backend Interface
 * Separates device-specific implementations from frontend
 * 
 * Design:
 * - All device-specific code goes in backend/
 * - Frontend (Tensor, Op, Autograd) is device-agnostic
 * - Can swap implementations (CPU ↔ GPU ↔ TPU) at runtime
 */

struct Backend {
    virtual ~Backend() = default;
    
    // Memory management
    virtual std::shared_ptr<float[]> allocate(size_t size) = 0;
    virtual void deallocate(std::shared_ptr<float[]> ptr) = 0;
    virtual void memcpy_h2d(float* device_ptr, const float* host_ptr, size_t size) = 0;
    virtual void memcpy_d2h(float* host_ptr, const float* device_ptr, size_t size) = 0;
    
    // Core operations (CPU)
    virtual void matmul(const Tensor& A, const Tensor& B, Tensor& C) = 0;
    virtual void matmul_add_bias(const Tensor& A, const Tensor& B, const Tensor& bias, Tensor& C) = 0;
    virtual void matmul_transpose_b(const Tensor& A, const Tensor& B, Tensor& C) = 0;
    
    // Unary operations
    virtual void softmax(const Tensor& x, Tensor& out, int axis) = 0;
    virtual void log_softmax(const Tensor& x, Tensor& out, int axis) = 0;
    virtual void relu(const Tensor& x, Tensor& out) = 0;
    virtual void gelu(const Tensor& x, Tensor& out) = 0;
    virtual void silu(const Tensor& x, Tensor& out) = 0;
    
    // Reduction operations
    virtual void sum(const Tensor& x, Tensor& out) = 0;
    virtual void mean(const Tensor& x, Tensor& out) = 0;
    
    virtual std::string name() const = 0;
};

// Global backend instance
Backend& get_backend();
void set_backend(std::unique_ptr<Backend> backend);

} // namespace backend
} // namespace aresml
