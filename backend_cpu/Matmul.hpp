#pragma once

#include "../core/Tensor.hpp"
#include "../utils/Profiler.hpp"
#include <cstring>

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

namespace aresml {
namespace backend_cpu {

constexpr size_t BLOCK_SIZE = 32;

inline void matmul_blocked(const Tensor& A, const Tensor& B, Tensor& C) {
    PROFILE_SCOPE("backend::matmul_blocked");
    
    if (A.shape.n != 2 || B.shape.n != 2) {
        throw std::runtime_error("matmul: both tensors must be 2D");
    }
    if (A.shape[1] != B.shape[0]) {
        throw std::runtime_error("matmul: dimension mismatch");
    }
    
    size_t M = A.shape[0];
    size_t K = A.shape[1];
    size_t N = B.shape[1];
    
    const float* a = A.begin();
    const float* b = B.begin();
    float* c = C.begin();
    
    for (size_t i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        for (size_t j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
            for (size_t k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                size_t i_max = (i0 + BLOCK_SIZE < M) ? i0 + BLOCK_SIZE : M;
                size_t j_max = (j0 + BLOCK_SIZE < N) ? j0 + BLOCK_SIZE : N;
                size_t k_max = (k0 + BLOCK_SIZE < K) ? k0 + BLOCK_SIZE : K;
                
                for (size_t i = i0; i < i_max; ++i) {
                    for (size_t j = j0; j < j_max; ++j) {
                        float sum = 0.0f;
                        for (size_t k = k0; k < k_max; ++k) {
                            sum += a[i * K + k] * b[k * N + j];
                        }
                        c[i * N + j] += sum;
                    }
                }
            }
        }
    }
}

inline void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    PROFILE_SCOPE("backend::matmul");
    
    if (A.shape.n != 2 || B.shape.n != 2) {
        throw std::runtime_error("matmul: both tensors must be 2D");
    }
    if (A.shape[1] != B.shape[0]) {
        throw std::runtime_error("matmul: dimension mismatch");
    }
    
#if defined(USE_OPENBLAS) && defined(OPENBLAS_AVAILABLE)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
               M, N, K, 1.0f, A.begin(), K, B.begin(), N, 0.0f, C.begin(), N);
#else
    matmul_blocked(A, B, C);
#endif
}

inline void matmul_add_bias(const Tensor& A, const Tensor& B, const Tensor& bias, Tensor& C) {
    PROFILE_SCOPE("backend::matmul_add_bias");
    
    if (A.shape.n != 2 || B.shape.n != 2) {
        throw std::runtime_error("matmul: both tensors must be 2D");
    }
    if (A.shape[1] != B.shape[0]) {
        throw std::runtime_error("matmul: dimension mismatch");
    }
    
    size_t M = A.shape[0];
    size_t K = A.shape[1];
    size_t N = B.shape[1];
    
    const float* a = A.begin();
    const float* b = B.begin();
    const float* bias_data = (bias.data) ? bias.begin() : nullptr;
    float* c = C.begin();
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[k * N + j];
            }
            if (bias_data) {
                sum += bias_data[j];
            }
            c[i * N + j] = sum;
        }
    }
}

inline void matmul_transpose_b(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.shape.n != 2 || B.shape.n != 2) {
        throw std::runtime_error("matmul: both tensors must be 2D");
    }
    if (A.shape[1] != B.shape[1]) {
        throw std::runtime_error("matmul: dimension mismatch");
    }
    
    size_t M = A.shape[0];
    size_t K = A.shape[1];
    size_t N = B.shape[0];
    
    const float* a = A.begin();
    const float* b = B.begin();
    float* c = C.begin();
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[j * K + k];
            }
            c[i * N + j] = sum;
        }
    }
}

inline Tensor matmul(const Tensor& A, const Tensor& B) {
    Tensor C({A.shape[0], B.shape[1]});
    matmul(A, B, C);
    return C;
}

}
}