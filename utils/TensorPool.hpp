#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <list>

namespace aresml {

/**
 * TensorPool - Reuses allocated buffers to avoid malloc/free in tight loops
 * 
 * Strategy:
 * - Maintain freelists for common tensor sizes
 * - On tensor deletion, return buffer to pool instead of freeing
 * - On tensor allocation, try pool first before malloc
 * - Reduces allocation overhead significantly in training loops
 * 
 * Note: Only pools small temporary tensors, not parameters
 */

class TensorPool {
private:
    static constexpr size_t MAX_POOL_SIZE = 100;  // max buffers per size
    static constexpr size_t MIN_POOL_BYTES = 64;   // min size to pool
    
    struct Bucket {
        std::list<std::shared_ptr<float[]>> free_list;
        size_t block_size = 0;  // bytes per block
        size_t allocated_count = 0;
        size_t reused_count = 0;
    };
    
    std::unordered_map<size_t, Bucket> pools;
    
    static TensorPool& instance() {
        static TensorPool pool;
        return pool;
    }
    
    size_t round_up_size(size_t bytes) {
        // Round to nearest power of 2 for better pooling
        if (bytes < MIN_POOL_BYTES) return MIN_POOL_BYTES;
        
        size_t rounded = 1;
        while (rounded < bytes) rounded *= 2;
        return rounded;
    }
    
public:
    TensorPool() = default;
    
    /**
     * Allocate a tensor buffer from pool if available, else malloc
     */
    static std::shared_ptr<float[]> allocate(size_t count) {
        size_t bytes = count * sizeof(float);
        size_t rounded_bytes = instance().round_up_size(bytes);
        size_t float_count = rounded_bytes / sizeof(float);
        
        Bucket& bucket = instance().pools[rounded_bytes];
        bucket.block_size = rounded_bytes;
        
        if (!bucket.free_list.empty()) {
            auto buffer = bucket.free_list.front();
            bucket.free_list.pop_front();
            bucket.reused_count++;
            return buffer;
        }
        
        bucket.allocated_count++;
        return std::shared_ptr<float[]>(new float[float_count]());
    }
    
    /**
     * Return a buffer to the pool for reuse
     */
    static void deallocate(std::shared_ptr<float[]> buffer, size_t count) {
        size_t bytes = count * sizeof(float);
        size_t rounded_bytes = instance().round_up_size(bytes);
        
        Bucket& bucket = instance().pools[rounded_bytes];
        
        if (bucket.free_list.size() < MAX_POOL_SIZE) {
            bucket.free_list.push_back(buffer);
        }
        // else: buffer is freed when shared_ptr goes out of scope
    }
    
    /**
     * Get pool statistics
     */
    struct Stats {
        size_t total_allocated = 0;
        size_t total_reused = 0;
        size_t buckets_count = 0;
    };
    
    static Stats get_stats() {
        Stats s;
        for (auto& [size, bucket] : instance().pools) {
            s.total_allocated += bucket.allocated_count;
            s.total_reused += bucket.reused_count;
            if (bucket.allocated_count > 0 || bucket.reused_count > 0) {
                s.buckets_count++;
            }
        }
        return s;
    }
    
    static void reset() {
        instance().pools.clear();
    }
};

} // namespace aresml
