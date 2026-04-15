#pragma once

#include <thread>
#include <vector>
#include <functional>
#include <atomic>

namespace aresml {
namespace threading {

// ============================================================================
// Thread pool and parallel utilities
// ============================================================================

class ParallelEngine {
public:
    static ParallelEngine& get() {
        static ParallelEngine instance;
        return instance;
    }
    
    void set_num_threads(size_t n) {
        num_threads_ = (n > 0) ? n : 1;
    }
    
    size_t get_num_threads() const {
        return num_threads_;
    }
    
    size_t hardware_concurrency() const {
        return std::thread::hardware_concurrency();
    }
    
    void parallel_for(size_t start, size_t end, size_t grain,
                      std::function<void(size_t, size_t)> fn) {
        if (end <= start || grain == 0) return;
        
        size_t n = end - start;
        
        // For small workloads, use single thread
        if (n < 10000) {
            fn(start, end);
            return;
        }
        
        size_t num_threads = num_threads_;
        if (num_threads > n) num_threads = n;
        
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        
        size_t chunk_size = (n + num_threads - 1) / num_threads;
        
        for (size_t t = 0; t < num_threads; ++t) {
            size_t chunk_start = start + t * chunk_size;
            if (chunk_start >= end) break;
            
            size_t chunk_end = chunk_start + chunk_size;
            if (chunk_end > end) chunk_end = end;
            
            threads.emplace_back([fn, chunk_start, chunk_end]() {
                fn(chunk_start, chunk_end);
            });
        }
        
        for (auto& th : threads) {
            th.join();
        }
    }
    
    // Simple parallel for with automatic grain
    void parallel_for(size_t start, size_t end, std::function<void(size_t, size_t)> fn) {
        size_t n = end - start;
        size_t grain = (n + num_threads_ - 1) / num_threads_;
        if (grain < 100) grain = 100;
        parallel_for(start, end, grain, fn);
    }
    
    // Thread-local accumulation for sum
    template<typename T>
    T parallel_reduce(size_t start, size_t end, 
                       std::function<T(size_t, size_t)> fn,
                       std::function<T(T, T)> combine) {
        if (end <= start) return T();
        
        size_t n = end - start;
        
        if (n < 10000) {
            return fn(start, end);
        }
        
        size_t num_threads = num_threads_;
        if (num_threads > n) num_threads = n;
        
        std::vector<T> results(num_threads);
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        
        size_t chunk_size = (n + num_threads - 1) / num_threads;
        
        for (size_t t = 0; t < num_threads; ++t) {
            size_t chunk_start = start + t * chunk_size;
            if (chunk_start >= end) break;
            
            size_t chunk_end = chunk_start + chunk_size;
            if (chunk_end > end) chunk_end = end;
            
            threads.emplace_back([fn, &results, t, chunk_start, chunk_end]() {
                results[t] = fn(chunk_start, chunk_end);
            });
        }
        
        for (auto& th : threads) {
            th.join();
        }
        
        T result = results[0];
        for (size_t i = 1; i < num_threads; ++i) {
            result = combine(result, results[i]);
        }
        
        return result;
    }

private:
    ParallelEngine() : num_threads_(std::thread::hardware_concurrency()) {
        if (num_threads_ == 0) num_threads_ = 1;
    }
    
    size_t num_threads_;
};

// Inline helper functions
inline void set_num_threads(size_t n) {
    ParallelEngine::get().set_num_threads(n);
}

inline size_t get_num_threads() {
    return ParallelEngine::get().get_num_threads();
}

inline void parallel_for(size_t start, size_t end, std::function<void(size_t, size_t)> fn) {
    ParallelEngine::get().parallel_for(start, end, fn);
}

} // namespace threading
} // namespace aresml