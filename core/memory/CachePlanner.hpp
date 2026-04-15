#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

namespace aresml {

struct CacheBlock {
    size_t start_row;
    size_t start_col;
    size_t block_m;
    size_t block_n;
    size_t block_k;
    size_t size_bytes;
    float reuse_factor;
    
    CacheBlock() 
        : start_row(0), start_col(0)
        , block_m(64), block_n(64), block_k(32)
        , size_bytes(0), reuse_factor(0.0f) {}
    
    CacheBlock(size_t m, size_t n, size_t k)
        : start_row(0), start_col(0)
        , block_m(m), block_n(n), block_k(k) {
        size_bytes = m * n * sizeof(float);
        reuse_factor = 1.0f;
    }
    
    size_t elements() const { return block_m * block_n; }
};

struct MemoryTile {
    void* data;
    size_t offset;
    size_t size;
    size_t alignment;
    
    MemoryTile() : data(nullptr), offset(0), size(0), alignment(64) {}
    MemoryTile(void* d, size_t off, size_t sz, size_t align = 64)
        : data(d), offset(off), size(sz), alignment(align) {}
};

class CachePlanner {
public:
    static constexpr size_t L1_CACHE_SIZE = 32 * 1024;
    static constexpr size_t L2_CACHE_SIZE = 256 * 1024;
    static constexpr size_t L3_CACHE_SIZE = 4 * 1024 * 1024;
    
    static constexpr size_t DEFAULT_BLOCK_M = 64;
    static constexpr size_t DEFAULT_BLOCK_N = 64;
    static constexpr size_t DEFAULT_BLOCK_K = 32;
    
    static constexpr size_t SIMD_WIDTH = 8;
    static constexpr size_t CACHE_LINE_SIZE = 64;
    
    static CacheBlock compute_optimal_block(size_t m, size_t n, size_t k) {
        CacheBlock block(DEFAULT_BLOCK_M, DEFAULT_BLOCK_N, DEFAULT_BLOCK_K);
        
        size_t l1_available = L1_CACHE_SIZE / sizeof(float);
        size_t l2_available = L2_CACHE_SIZE / sizeof(float);
        
        if (m * k < l1_available / 2) {
            block.block_m = m;
        } else {
            block.block_m = DEFAULT_BLOCK_M;
        }
        
        if (k * n < l1_available / 2) {
            block.block_k = k;
        } else {
            block.block_k = DEFAULT_BLOCK_K;
        }
        
        if (n < l1_available / 4) {
            block.block_n = n;
        } else {
            block.block_n = DEFAULT_BLOCK_N;
        }
        
        block.size_bytes = block.block_m * block.block_n * sizeof(float);
        block.reuse_factor = compute_reuse_factor(block, m, n, k);
        
        return block;
    }
    
    static std::vector<CacheBlock> generate_blocking_plan(size_t m, size_t n, size_t k) {
        std::vector<CacheBlock> blocks;
        
        CacheBlock optimal = compute_optimal_block(m, n, k);
        
        for (size_t i = 0; i < m; i += optimal.block_m) {
            for (size_t j = 0; j < n; j += optimal.block_n) {
                CacheBlock block = optimal;
                block.start_row = i;
                block.start_col = j;
                
                size_t actual_m = (i + optimal.block_m > m) ? (m - i) : optimal.block_m;
                size_t actual_n = (j + optimal.block_n > n) ? (n - j) : optimal.block_n;
                
                block.block_m = actual_m;
                block.block_n = actual_n;
                block.size_bytes = actual_m * actual_n * sizeof(float);
                
                blocks.push_back(block);
            }
        }
        
        return blocks;
    }
    
    static size_t compute_prefetch_distance(const CacheBlock& block) {
        size_t bytes = block.size_bytes;
        size_t prefetch_distance = 2;
        
        if (bytes > L1_CACHE_SIZE / 2) {
            prefetch_distance = 1;
        } else if (bytes < L1_CACHE_SIZE / 8) {
            prefetch_distance = 4;
        }
        
        return prefetch_distance;
    }
    
    static float estimate_bandwidth_reduction(const CacheBlock& block, size_t m, size_t n, size_t k) {
        size_t original_loads = m * k + k * n;
        size_t blocked_ops = ((m + block.block_m - 1) / block.block_m) *
                             ((n + block.block_n - 1) / block.block_n);
        
        float reduction = 1.0f;
        if (blocked_ops > 0) {
            float blocked_loads = static_cast<float>(blocked_ops * block.block_k);
            reduction = static_cast<float>(original_loads) / blocked_loads;
        }
        
        return reduction;
    }
    
private:
    static float compute_reuse_factor(const CacheBlock& block, size_t m, size_t n, size_t k) {
        size_t total_blocks = ((m + block.block_m - 1) / block.block_m) *
                              ((n + block.block_n - 1) / block.block_n);
        
        if (total_blocks == 0) return 0.0f;
        
        float reuse = static_cast<float>(block.block_k) / static_cast<float>(total_blocks);
        return reuse;
    }
};

inline size_t align_to_cache_line(size_t size) {
    size_t alignment = CachePlanner::CACHE_LINE_SIZE;
    return (size + alignment - 1) & ~(alignment - 1);
}

inline size_t get_cache_level(size_t size_bytes) {
    if (size_bytes <= CachePlanner::L1_CACHE_SIZE) return 1;
    if (size_bytes <= CachePlanner::L2_CACHE_SIZE) return 2;
    if (size_bytes <= CachePlanner::L3_CACHE_SIZE) return 3;
    return 0;
}

}
