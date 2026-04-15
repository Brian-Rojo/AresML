#pragma once

#include "Tensor.hpp"
#include <unordered_map>
#include <unordered_set>
#include <mutex>

namespace aresml {

class SafeTensorPool {
public:
    struct Entry {
        Tensor* tensor;
        int in_use_count;
        bool locked_by_graph;
        uint64_t generation;
        
        Entry() : tensor(nullptr), in_use_count(0), locked_by_graph(false), generation(0) {}
    };
    
    static Tensor* acquire(const Shape& shape) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        auto key = shape.size();
        auto it = free_list_.find(key);
        
        if (it != free_list_.end() && !it->second.empty()) {
            Entry& entry = it->second.back();
            
            if (entry.locked_by_graph) {
                std::cerr << "[SAFE_POOL] WARNING: trying to reuse locked tensor!\n";
                return new Tensor(shape, false);
            }
            
            Tensor* t = entry.tensor;
            entry.in_use_count++;
            
            it->second.pop_back();
            active_tensors_[t] = entry;
            
            t->offset = 0;
            if (t->grad) {
                t->grad->zero_();
            }
            
            return t;
        }
        
        Tensor* new_t = new Tensor(shape, false);
        Entry new_entry;
        new_entry.tensor = new_t;
        new_entry.in_use_count = 1;
        new_entry.generation = current_generation_;
        active_tensors_[new_t] = new_entry;
        
        return new_t;
    }
    
    static void release(Tensor* t) {
        if (!t) return;
        
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        auto it = active_tensors_.find(t);
        if (it != active_tensors_.end()) {
            if (it->second.locked_by_graph) {
                std::cerr << "[SAFE_POOL] WARNING: releasing locked tensor!\n";
            }
            
            it->second.in_use_count--;
            if (it->second.in_use_count == 0 && !it->second.locked_by_graph) {
                free_list_[t->shape.size()].push_back(it->second);
                active_tensors_.erase(it);
            }
            return;
        }
        
        delete t;
    }
    
    static void lock_for_graph(Tensor* t) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        auto it = active_tensors_.find(t);
        if (it != active_tensors_.end()) {
            it->second.locked_by_graph = true;
        }
    }
    
    static void unlock_from_graph(Tensor* t) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        auto it = active_tensors_.find(t);
        if (it != active_tensors_.end()) {
            it->second.locked_by_graph = false;
        }
    }
    
    static void increment_generation() {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        current_generation_++;
    }
    
    static bool is_valid(const Tensor* t) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        auto it = active_tensors_.find(const_cast<Tensor*>(t));
        if (it != active_tensors_.end()) {
            return it->second.generation == current_generation_;
        }
        
        return true;
    }
    
    static void clear() {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        for (auto& pair : free_list_) {
            for (auto& entry : pair.second) {
                delete entry.tensor;
            }
            pair.second.clear();
        }
        
        for (auto& pair : active_tensors_) {
            if (pair.second.tensor) {
                delete pair.second.tensor;
            }
        }
        
        free_list_.clear();
        active_tensors_.clear();
    }
    
    static size_t active_count() {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        return active_tensors_.size();
    }
    
    static size_t pool_size() {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        
        size_t total = 0;
        for (auto& pair : free_list_) {
            total += pair.second.size();
        }
        return total;
    }

private:
    static std::unordered_map<size_t, std::vector<Entry>> free_list_;
    static std::unordered_map<Tensor*, Entry> active_tensors_;
    static std::mutex pool_mutex_;
    static uint64_t current_generation_;
};

std::unordered_map<size_t, std::vector<SafeTensorPool::Entry>> SafeTensorPool::free_list_;
std::unordered_map<Tensor*, SafeTensorPool::Entry> SafeTensorPool::active_tensors_;
std::mutex SafeTensorPool::pool_mutex_;
uint64_t SafeTensorPool::current_generation_ = 0;

inline Tensor* safe_alloc(const Shape& shape) {
    return SafeTensorPool::acquire(shape);
}

inline void safe_free(Tensor* t) {
    SafeTensorPool::release(t);
}

}