#pragma once

#include "Backend.hpp"

namespace aresml {
namespace backend {

/**
 * CPU Backend Implementation
 * Uses naive CPU kernels (suitable for testing, not production)
 * For better performance in production, use:
 * - OpenBLAS, Intel MKL for matmul
 * - SIMD(AVX2) for element-wise operations
 * - Custom kernels with better cache locality
 */

class CPUBackend : public Backend {
public:
    // Memory management (simple malloc/free)
    std::shared_ptr<float[]> allocate(size_t size) override;
    void deallocate(std::shared_ptr<float[]> ptr) override;
    void memcpy_h2d(float* device_ptr, const float* host_ptr, size_t size) override;
    void memcpy_d2h(float* host_ptr, const float* device_ptr, size_t size) override;
    
    // Core operations
    void matmul(const Tensor& A, const Tensor& B, Tensor& C) override;
    void matmul_add_bias(const Tensor& A, const Tensor& B, const Tensor& bias, Tensor& C) override;
    void matmul_transpose_b(const Tensor& A, const Tensor& B, Tensor& C) override;
    
    // Unary operations
    void softmax(const Tensor& x, Tensor& out, int axis) override;
    void log_softmax(const Tensor& x, Tensor& out, int axis) override;
    void relu(const Tensor& x, Tensor& out) override;
    void gelu(const Tensor& x, Tensor& out) override;
    void silu(const Tensor& x, Tensor& out) override;
    
    // Reduction operations
    void sum(const Tensor& x, Tensor& out) override;
    void mean(const Tensor& x, Tensor& out) override;
    
    std::string name() const override { return "cpu"; }
};

} // namespace backend
} // namespace aresml
