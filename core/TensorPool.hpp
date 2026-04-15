#pragma once

#include "Tensor.hpp"
#include <vector>
#include <mutex>
#include <unordered_map>

namespace aresml {

constexpr size_t MAX_BATCH_SIZE = 16;
constexpr size_t MAX_SEQ_LEN = 32;
constexpr size_t MAX_HIDDEN = 128;

class TensorPool {
public:
    static Tensor* acquire(const Shape& shape) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        auto key = shape.size();
        auto it = free_list_.find(key);
        
        if (it != free_list_.end() && !it->second.empty()) {
            Tensor* t = it->second.back();
            it->second.pop_back();
            t->offset = 0;
            if (t->grad) {
                t->grad->zero_();
            }
            return t;
        }
        
        return new Tensor(shape, false);
    }
    
    static void release(Tensor* t) {
        if (!t) return;
        
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        auto key = t->shape.size();
        free_list_[key].push_back(t);
    }
    
    static void reserve(size_t n_tensors, const Shape& common_shape) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        size_t key = common_shape.size();
        free_list_[key].reserve(n_tensors);
        
        for (size_t i = 0; i < n_tensors; ++i) {
            free_list_[key].push_back(new Tensor(common_shape, false));
        }
    }
    
    static void clear() {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        for (auto& pair : free_list_) {
            for (Tensor* t : pair.second) {
                delete t;
            }
            pair.second.clear();
        }
        free_list_.clear();
    }
    
    static size_t total_allocated() {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        size_t total = 0;
        for (auto& pair : free_list_) {
            total += pair.second.size();
        }
        return total;
    }
    
    static size_t pool_size() {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        return free_list_.size();
    }

private:
    static std::unordered_map<size_t, std::vector<Tensor*>> free_list_;
    static std::mutex pool_mutex_;
};

std::unordered_map<size_t, std::vector<Tensor*>> TensorPool::free_list_;
std::mutex TensorPool::pool_mutex_;

inline Tensor* tensor_alloc(const Shape& shape) {
    return TensorPool::acquire(shape);
}

inline void tensor_free(Tensor* t) {
    TensorPool::release(t);
}

}