#pragma once

#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace aresml {

enum class PrefetchHint {
    NONE,
    READ_L1,
    READ_L2,
    READ_L3,
    WRITE_L1,
    WRITE_L2,
    WRITE_L3
};

struct PrefetchRequest {
    void* address;
    size_t size;
    PrefetchHint hint;
    int priority;
    
    PrefetchRequest() 
        : address(nullptr), size(0), hint(PrefetchHint::NONE), priority(0) {}
    
    PrefetchRequest(void* addr, size_t sz, PrefetchHint h = PrefetchHint::READ_L2)
        : address(addr), size(sz), hint(h), priority(0) {}
};

class PrefetchEngine {
public:
    static constexpr size_t DEFAULT_PREFETCH_DISTANCE = 2;
    static constexpr size_t MAX_PREFETCH_QUEUE = 16;
    
    PrefetchEngine() : enabled_(true), distance_(DEFAULT_PREFETCH_DISTANCE) {}
    
    explicit PrefetchEngine(size_t distance) : enabled_(true), distance_(distance) {}
    
    void prefetch(const PrefetchRequest& req) {
        if (!enabled_ || !req.address) return;
        prefetch_l3(req.address);
    }
    
    void prefetch_array(float* data, size_t count, size_t stride = 1) {
        if (!enabled_ || !data) return;
        
        size_t prefetch_count = count > distance_ ? distance_ : count;
        
        for (size_t i = 0; i < prefetch_count; i += stride) {
            prefetch_l3(&data[i]);
        }
    }
    
    void prefetch_matmul_block(float* a, float* b, size_t m, size_t n, size_t k,
                               size_t row, size_t col) {
        if (!enabled_) return;
        
        size_t block_m = 64;
        size_t block_n = 64;
        
        for (size_t i = 0; i < block_m && row + i < m; i += 4) {
            for (size_t j = 0; j < block_n && col + j < n; j += 4) {
                prefetch_l3(&a[(row + i) * k + col]);
                prefetch_l3(&b[(row + i) * n + (col + j)]);
            }
        }
    }
    
    void prefetch_next_block(float* a, float* b, size_t m, size_t n, size_t k,
                             size_t current_row, size_t current_col,
                             size_t block_m = 64, size_t block_n = 64) {
        if (!enabled_) return;
        
        size_t next_row = current_row + block_m;
        size_t next_col = current_col + block_n;
        
        if (next_row < m) {
            size_t prefetch_m = std::min(block_m, m - next_row);
            prefetch_array(&a[next_row * k], prefetch_m * k, 1);
        }
        
        if (next_col < n) {
            size_t prefetch_n = std::min(block_n, n - next_col);
            prefetch_array(&b[current_row * n + next_col], prefetch_n, 1);
        }
    }
    
    void prefetch_elementwise(float* data, size_t count) {
        if (!enabled_ || !data) return;
        
        size_t prefetch_dist = count > distance_ ? distance_ : count;
        
        for (size_t i = 0; i < prefetch_dist; ++i) {
            prefetch_l3(&data[i]);
        }
    }
    
    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool is_enabled() const { return enabled_; }
    
    void set_distance(size_t distance) { distance_ = distance; }
    size_t get_distance() const { return distance_; }
    
private:
    bool enabled_;
    size_t distance_;
    
    static inline void prefetch_l3(void* addr) {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(addr, 0, 3);
#endif
    }
    
    static inline void prefetch_l2(void* addr) {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(addr, 0, 2);
#endif
    }
    
    static inline void prefetch_l1(void* addr) {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(addr, 0, 1);
#endif
    }
};

inline void prefetch_read(void* addr) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(addr, 0, 3);
#endif
}

inline void prefetch_write(void* addr) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(addr, 1, 3);
#endif
}

}
