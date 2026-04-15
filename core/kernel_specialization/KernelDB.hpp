#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>
#include "../ir_v3/IRGraphV3.hpp"
#include "../../core/Tensor.hpp"

namespace aresml {

enum class KernelShape {
    SMALL,
    MEDIUM,
    LARGE,
    HUGE
};

struct KernelSpec {
    std::string name;
    KernelShape shape;
    size_t m, n, k;
    bool use_simd;
    bool use_prefetch;
    bool use_blocking;
    size_t block_m, block_n, block_k;
    size_t estimated_cycles;
    float measured_time_ms;
    
    KernelSpec() : name("default"), shape(KernelShape::MEDIUM), m(0), n(0), k(0),
                  use_simd(true), use_prefetch(true), use_blocking(true),
                  block_m(64), block_n(64), block_k(32), estimated_cycles(0), measured_time_ms(0.0f) {}
    
    KernelSpec(const std::string& n, KernelShape s, size_t M, size_t N, size_t K,
               bool simd, bool prefetch, bool block, size_t bm, size_t bn, size_t bk,
               size_t cycles, float time)
        : name(n), shape(s), m(M), n(N), k(K),
          use_simd(simd), use_prefetch(prefetch), use_blocking(block),
          block_m(bm), block_n(bn), block_k(bk), estimated_cycles(cycles), measured_time_ms(time) {}
    
    std::string signature() const {
        return name + "_" + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
    }
};

class KernelDB {
public:
    static KernelSpec get_kernel(IROpV3 op, const std::vector<int64_t>& shape) {
        if (op == IROpV3::GEMM || op == IROpV3::GEMM_BIAS_RELU) {
            return get_gemm_kernel(shape);
        } else if (op == IROpV3::ELEMENTWISE_ADD || op == IROpV3::ELEMENTWISE_MUL) {
            return get_elementwise_kernel(op, shape);
        } else if (op == IROpV3::ELEMENTWISE_RELU || op == IROpV3::ELEMENTWISE_GELU) {
            return get_activation_kernel(op, shape);
        }
        
        return KernelSpec{"generic", KernelShape::MEDIUM, shape[0], shape[1], 0, true, true, true, 64, 64, 32, 0, 0.0f};
    }
    
private:
    static KernelSpec get_gemm_kernel(const std::vector<int64_t>& shape) {
        if (shape.size() < 2) {
            return KernelSpec{"gemm_small", KernelShape::SMALL, shape[0], shape[1], 0, true, false, false, 16, 16, 8, 0, 0.0f};
        }
        
        size_t m = shape[0];
        size_t n = shape[1];
        size_t k = shape.size() > 2 ? shape[2] : m;
        
        if (m <= 32 && n <= 32) {
            return KernelSpec{"gemm_32x32", KernelShape::SMALL, m, n, k, true, false, false, 16, 16, 8, 0, 0.0f};
        } else if (m <= 64 && n <= 64) {
            return KernelSpec{"gemm_64x64", KernelShape::MEDIUM, m, n, k, true, true, true, 64, 64, 32, 0, 0.0f};
        } else if (m <= 128 && n <= 128) {
            return KernelSpec{"gemm_128x128", KernelShape::LARGE, m, n, k, true, true, true, 64, 64, 32, 0, 0.0f};
        } else {
            return KernelSpec{"gemm_256x256", KernelShape::HUGE, m, n, k, true, true, true, 64, 64, 32, 0, 0.0f};
        }
    }
    
    static KernelSpec get_elementwise_kernel(IROpV3 op, const std::vector<int64_t>& shape) {
        size_t n = 1;
        for (auto d : shape) n *= d;
        
        return KernelSpec{op == IROpV3::ELEMENTWISE_ADD ? "elem_add" : "elem_mul",
                         n < 1024 ? KernelShape::SMALL : KernelShape::MEDIUM,
                         1, n, 1, true, false, false, 64, 64, 32, 0, 0.0f};
    }
    
    static KernelSpec get_activation_kernel(IROpV3 op, const std::vector<int64_t>& shape) {
        size_t n = 1;
        for (auto d : shape) n *= d;
        
        return KernelSpec{op == IROpV3::ELEMENTWISE_RELU ? "relu" : "gelu",
                         KernelShape::MEDIUM, 1, n, 1, true, false, false, 64, 64, 32, 0, 0.0f};
    }
};

class KernelSelector {
public:
    static KernelSpec select(const std::vector<KernelSpec>& candidates, const std::string& target_device) {
        KernelSpec best = candidates[0];
        
        for (const auto& cand : candidates) {
            if (cand.measured_time_ms > 0 && 
                (best.measured_time_ms == 0 || cand.measured_time_ms < best.measured_time_ms)) {
                best = cand;
            }
        }
        
        return best;
    }
    
    static std::vector<KernelSpec> get_candidates(IROpV3 op, const std::vector<int64_t>& shape) {
        std::vector<KernelSpec> candidates;
        
        candidates.push_back(KernelDB::get_kernel(op, shape));
        
        if (op == IROpV3::GEMM) {
            candidates.push_back(KernelSpec{"gemm_avx2", KernelShape::MEDIUM, shape[0], shape[1], 
                                          shape.size() > 2 ? shape[2] : shape[0],
                                          true, true, true, 64, 64, 32, 0, 0.0f});
            candidates.push_back(KernelSpec{"gemm_blocked", KernelShape::MEDIUM, shape[0], shape[1],
                                           shape.size() > 2 ? shape[2] : shape[0],
                                           true, true, true, 32, 32, 16, 0, 0.0f});
        }
        
        return candidates;
    }
};

}
