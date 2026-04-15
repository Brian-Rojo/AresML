#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstddef>
#include <cstdint>

namespace aresml {

class CacheBlockingV2 {
public:
    static constexpr size_t L1_SIZE = 32 * 1024;
    static constexpr size_t L2_SIZE = 256 * 1024;
    static constexpr size_t L3_SIZE = 4 * 1024 * 1024;
    
    static constexpr size_t DEFAULT_BLOCK_M = 64;
    static constexpr size_t DEFAULT_BLOCK_N = 64;
    static constexpr size_t DEFAULT_BLOCK_K = 32;
    
    struct BlockConfig {
        size_t block_m, block_n, block_k;
        size_t inner_m, inner_n;
        bool use_prefetch;
        size_t prefetch_distance;
        
        BlockConfig() : block_m(64), block_n(64), block_k(32), inner_m(8), inner_n(8),
                       use_prefetch(true), prefetch_distance(2) {}
    };
    
    static BlockConfig compute_blocking(size_t m, size_t n, size_t k) {
        BlockConfig config;
        
        size_t total_elements = m * n;
        
        if (total_elements <= 1024) {
            config.block_m = 16;
            config.block_n = 16;
            config.block_k = 8;
            config.inner_m = 4;
            config.inner_n = 4;
            config.use_prefetch = false;
        } else if (total_elements <= 4096) {
            config.block_m = 32;
            config.block_n = 32;
            config.block_k = 16;
            config.inner_m = 8;
            config.inner_n = 8;
            config.use_prefetch = true;
            config.prefetch_distance = 1;
        } else if (total_elements <= 16384) {
            config.block_m = 64;
            config.block_n = 64;
            config.block_k = 32;
            config.inner_m = 8;
            config.inner_n = 8;
            config.use_prefetch = true;
            config.prefetch_distance = 2;
        } else {
            config.block_m = 64;
            config.block_n = 64;
            config.block_k = 32;
            config.inner_m = 8;
            config.inner_n = 8;
            config.use_prefetch = true;
            config.prefetch_distance = 2;
        }
        
        return config;
    }
    
    static std::vector<BlockConfig> generate_multi_level_blocks(size_t m, size_t n, size_t k) {
        std::vector<BlockConfig> configs;
        
        configs.push_back(compute_blocking(m, n, k));
        
        if (m > 128 || n > 128) {
            BlockConfig outer;
            outer.block_m = 128;
            outer.block_n = 128;
            outer.block_k = 64;
            outer.inner_m = 16;
            outer.inner_n = 16;
            outer.use_prefetch = true;
            outer.prefetch_distance = 1;
            configs.push_back(outer);
        }
        
        return configs;
    }
};

class StrideOptimizer {
public:
    static void optimize_layout(std::vector<int64_t>& strides, const std::vector<int64_t>& shape) {
        if (shape.empty()) return;
        
        size_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }
    
    static bool is_contiguous(const std::vector<int64_t>& strides, const std::vector<int64_t>& shape) {
        if (shape.empty()) return true;
        
        size_t expected_stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            if (strides[i] != static_cast<int64_t>(expected_stride)) {
                return false;
            }
            expected_stride *= shape[i];
        }
        return true;
    }
    
    static std::vector<int64_t> compute_optimal_strides(const std::vector<int64_t>& shape) {
        std::vector<int64_t> strides(shape.size());
        optimize_layout(strides, shape);
        return strides;
    }
};

class LayoutAnalyzer {
public:
    struct TensorLayout {
        std::vector<int64_t> shape;
        std::vector<int64_t> strides;
        bool is_contiguous;
        size_t alignment;
        
        TensorLayout() : is_contiguous(true), alignment(64) {}
    };
    
    static TensorLayout analyze(const std::vector<int64_t>& shape) {
        TensorLayout layout;
        layout.shape = shape;
        layout.strides = StrideOptimizer::compute_optimal_strides(shape);
        layout.is_contiguous = StrideOptimizer::is_contiguous(layout.strides, shape);
        
        size_t total = 1;
        for (auto d : shape) total *= d;
        layout.alignment = (total * sizeof(float) % 64 == 0) ? 64 : 32;
        
        return layout;
    }
    
    static bool needs_reorder(const TensorLayout& layout) {
        return !layout.is_contiguous || layout.alignment < 64;
    }
};

}
