#pragma once

#include "../core/Tensor.hpp"
#include "../core/Autograd.hpp"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace aresml {

enum class MemoryStrategy {
    STORE_ALL,
    RECOMPUTE_ALL,
    HYBRID_OPTIMAL
};

class MemoryManager {
public:
    static bool strict_gc;
    static bool auto_cleanup;
    static MemoryStrategy strategy;
    
    static void enable(bool enable) {
        auto_cleanup = enable;
        if (enable) {
            std::cout << "[MEMORY] GC enabled\n";
        }
    }
    
    static void set_strategy(MemoryStrategy s) {
        strategy = s;
        std::cout << "[MEMORY] Strategy: ";
        switch(s) {
            case MemoryStrategy::STORE_ALL: std::cout << "STORE_ALL\n"; break;
            case MemoryStrategy::RECOMPUTE_ALL: std::cout << "RECOMPUTE_ALL\n"; break;
            case MemoryStrategy::HYBRID_OPTIMAL: std::cout << "HYBRID_OPTIMAL\n"; break;
        }
    }
    
    static void collect_garbage() {
        if (!auto_cleanup) return;
        
        auto& ctx = get_engine();
        
        ctx.collect_garbage();
        
        if (strict_gc) {
            std::cout << "[MEMORY] GC collected\n";
        }
    }
    
    static size_t get_peak_memory() {
        return peak_memory_;
    }
    
    static void reset_peak() {
        peak_memory_ = 0;
        std::cout << "[MEMORY] Peak reset\n";
    }
    
private:
    static size_t peak_memory_;
};

bool MemoryManager::strict_gc = false;
bool MemoryManager::auto_cleanup = false;
MemoryStrategy MemoryManager::strategy = MemoryStrategy::HYBRID_OPTIMAL;
size_t MemoryManager::peak_memory_ = 0;

class MicroKernelEngine {
public:
    static void fused_linear_relu(float* input, float* weight, float* output,
                                   size_t M, size_t K, size_t N) {
        if (!input || !weight || !output) return;
        
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += input[i * K + k] * weight[k * N + j];
                }
                output[i * N + j] = sum > 0.0f ? sum : 0.0f;
            }
        }
    }
    
    static void fused_gemm_bias(float* A, float* B, float* bias, float* C,
                               size_t M, size_t K, size_t N) {
        if (!A || !B || !C) return;
        
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = bias ? bias[j] : 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
    
    static void fused_gemm_bias_relu(float* A, float* B, float* bias, float* C,
                                     size_t M, size_t K, size_t N) {
        if (!A || !B || !C) return;
        
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = bias ? bias[j] : 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum > 0.0f ? sum : 0.0f;
            }
        }
    }
    
    static void vectorized_relu(float* data, size_t n) {
        if (!data) return;
        
        for (size_t i = 0; i < n; ++i) {
            data[i] = data[i] > 0.0f ? data[i] : 0.0f;
        }
    }
    
    static void layer_norm(float* data, float* mean, float* var, size_t n, float eps = 1e-5f) {
        if (!data || !mean || !var) return;
        
        float m = 0.0f;
        for (size_t i = 0; i < n; ++i) m += data[i];
        m /= n;
        
        float v = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            float diff = data[i] - m;
            v += diff * diff;
        }
        v /= n;
        
        float std = std::sqrt(v + eps);
        
        for (size_t i = 0; i < n; ++i) {
            data[i] = (data[i] - m) / std;
        }
        
        *mean = m;
        *var = v;
    }
};

class TrainingEngine {
public:
    static int current_step_;
    static int max_steps_before_checkpoint_;
    static float gradient_clip_value_;
    
    static void enable_autosave(int steps) {
        max_steps_before_checkpoint_ = steps;
        std::cout << "[TRAINING] autosave every " << steps << " steps\n";
    }
    
    static void set_gradient_clip(float value) {
        gradient_clip_value_ = value;
        if (value > 0) {
            std::cout << "[TRAINING] gradient clip: " << value << "\n";
        }
    }
    
    static void check_nan_gradients(Tensor* t, const std::string& name) {
        if (!t || !t->data) return;
        
        float* data = t->data.get() + t->offset;
        size_t n = t->shape.size();
        
        for (size_t i = 0; i < n; ++i) {
            if (std::isnan(data[i]) || std::isinf(data[i])) {
                std::cerr << "[TRAINING] NaN/Inf detected in " << name 
                         << " at index " << i << "\n";
                std::cerr << "[TRAINING] RECOVERING: resetting gradients\n";
                if (t->grad && t->grad->data) {
                    t->grad->zero_();
                }
                return;
            }
        }
    }
    
    static void step(Tensor& loss, float lr) {
        current_step_++;
        
        if (gradient_clip_value_ > 0) {
            clip_gradients(gradient_clip_value_);
        }
        
        if (current_step_ % 100 == 0) {
            std::cout << "[TRAINING] Step " << current_step_ 
                     << ", loss: " << (loss.data ? loss.data[0] : 0) << "\n";
        }
    }
    
private:
    static void clip_gradients(float max_norm) {
        auto& ctx = get_engine();
        
        for (auto* leaf : ctx.get_leaf_tensors()) {
            if (!leaf || !leaf->grad || !leaf->grad->data) continue;
            
            float* g = leaf->grad->data.get() + leaf->grad->offset;
            size_t n = leaf->grad->shape.size();
            
            float total_norm = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                total_norm += g[i] * g[i];
            }
            total_norm = std::sqrt(total_norm);
            
            if (total_norm > max_norm) {
                float scale = max_norm / total_norm;
                for (size_t i = 0; i < n; ++i) {
                    g[i] *= scale;
                }
            }
        }
    }
};

int TrainingEngine::current_step_ = 0;
int TrainingEngine::max_steps_before_checkpoint_ = 1000;
float TrainingEngine::gradient_clip_value_ = 0.0f;

class ExecutionScheduler {
public:
    static int num_threads() {
        return std::thread::hardware_concurrency();
    }
    
    static void print_info() {
        int n = num_threads();
        std::cout << "[SCHEDULER] Available threads: " << n << "\n";
        std::cout << "[SCHEDULER] Recommended workers: " << std::max(1, n - 1) << "\n";
    }
    
    static void set_threads(int n) {
        std::cout << "[SCHEDULER] Using " << n << " threads\n";
    }
};

inline void enable_memory_gc(bool strict = false) {
    MemoryManager::enable(true);
    MemoryManager::strict_gc = strict;
}

inline void set_memory_strategy(const std::string& s) {
    if (s == "store") {
        MemoryManager::set_strategy(MemoryStrategy::STORE_ALL);
    } else if (s == "recompute") {
        MemoryManager::set_strategy(MemoryStrategy::RECOMPUTE_ALL);
    } else {
        MemoryManager::set_strategy(MemoryStrategy::HYBRID_OPTIMAL);
    }
}

inline void enable_gradient_clip(float value) {
    TrainingEngine::set_gradient_clip(value);
}

inline void set_training_autosave(int steps) {
    TrainingEngine::enable_autosave(steps);
}

}