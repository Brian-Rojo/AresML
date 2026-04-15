#pragma once

#include "Tensor.hpp"
#include <unordered_set>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <queue>
#include <functional>
#include <memory>
#include <thread>
#include <unordered_map>

namespace aresml {

constexpr bool ARESML_DEBUG = false;

// ============================================================================
// ExecutionContext - Global autograd engine with parameter registry
// ============================================================================

class ExecutionContext {
private:
    // PARAMETER REGISTRY - Tracks ALL model parameters globally
    std::vector<Tensor*> parameters_;
    
    // Leaf tensor registry (for zero_grad and gradient clipping)
    std::unordered_set<Tensor*> leaf_tensors_;
    
    bool debug_mode_ = false;
    std::vector<Tensor*> backward_order_;
    std::string name_;
    bool accumulate_gradients_ = false;
    int accumulation_steps_ = 1;
    int current_step_ = 0;
    bool strict_mode_ = false;

public:
    ExecutionContext(const std::string& name = "default") : name_(name) {}

    void set_debug(bool d) { debug_mode_ = d; }
    bool is_debug() const { return debug_mode_; }

    const std::string& name() const { return name_; }

    void set_gradient_accumulation(bool enable, int steps = 1) {
        accumulate_gradients_ = enable;
        accumulation_steps_ = steps;
        current_step_ = 0;
    }

    bool is_accumulating() const { return accumulate_gradients_; }
    int accumulation_steps() const { return accumulation_steps_; }
    int current_step() const { return current_step_; }

    void set_strict_mode(bool enable) {
        strict_mode_ = enable;
        if (strict_mode_) {
            std::cout << "[STRICT] Strict mode enabled for: " << name_ << "\n";
        }
    }

    bool is_strict() const { return strict_mode_; }

    // =========================================================================
    // PARAMETER REGISTRY - CRITICAL: This replaces the broken leaf_tensors approach
    // =========================================================================

    /// Register a parameter tensor (e.g., weight, bias) in the global registry
    void register_parameter(Tensor* param) {
        if (!param) return;
        
        // Avoid duplicates
        for (auto* p : parameters_) {
            if (p == param) return;
        }
        
        parameters_.push_back(param);
        
        // Also register as leaf if it's a leaf tensor
        if (param->requires_grad && param->is_leaf) {
            leaf_tensors_.insert(param);
        }
        
        if (debug_mode_) {
            std::cout << "[PARAM] Registered parameter: shape=" << param->shape.size() 
                      << " requires_grad=" << param->requires_grad << "\n";
        }
    }

    /// Get all registered parameters
    const std::vector<Tensor*>& get_parameters() const {
        return parameters_;
    }

    /// Clear all parameters (e.g., between models)
    void clear_parameters() {
        parameters_.clear();
    }

    // =========================================================================
    // LEAF TENSOR REGISTRY
    // =========================================================================

    void register_leaf(Tensor* t) {
        if (t && t->requires_grad && t->is_leaf) {
            leaf_tensors_.insert(t);
            if (debug_mode_) {
                std::cout << "[LEAF] Registered leaf tensor: shape="
                          << t->shape.size() << " requires_grad=" << t->requires_grad << "\n";
            }
        }
    }

    // =========================================================================
    // GRADIENT CLIPPING
    // =========================================================================

    void clip_grad(float max_norm) {
        if (max_norm <= 0.0f) return;

        for (auto* t : parameters_) {
            if (!t || !t->grad || !t->grad->data) continue;

            float* g = t->grad->data.get() + t->grad->offset;
            size_t n = t->grad->shape.size();

            float total_norm = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                total_norm += g[i] * g[i];
            }
            total_norm = std::sqrt(total_norm);

            if (total_norm > max_norm) {
                float clip_factor = max_norm / (total_norm + EPSILON);
                for (size_t i = 0; i < n; ++i) {
                    g[i] *= clip_factor;
                }

                if (debug_mode_) {
                    std::cout << "[CLIP] Clipped gradient: " << total_norm << " -> " << max_norm << "\n";
                }
            }
        }
    }

    // =========================================================================
    // BACKWARD ENGINE - Global DAG traversal from loss to ALL parameters
    // =========================================================================

    void backward(Tensor* loss) {
        if (!loss) {
            throw std::runtime_error("backward: loss is null");
        }
        if (!loss->requires_grad) {
            throw std::runtime_error("backward: loss tensor does not require gradients");
        }

        if (debug_mode_) {
            std::cout << "\n=== BACKWARD START ===" << std::endl;
            std::cout << "Loss tensor: shape=" << loss->shape.size() 
                      << " requires_grad=" << loss->requires_grad << "\n";
            std::cout << "Registered parameters: " << parameters_.size() << "\n";
        }

        // STEP 1: Build backward order via topological sort from loss
        backward_order_.clear();
        std::unordered_set<Tensor*> visited;
        std::unordered_set<Tensor*> visiting;

        std::function<void(Tensor*)> dfs = [&](Tensor* node) {
            if (!node || !node->requires_grad) return;
            if (visiting.count(node)) {
                if (debug_mode_) {
                    std::cout << "[WARN] Cycle detected at tensor shape=" << node->shape.size() << "\n";
                }
                return;
            }
            if (visited.count(node)) return;

            visiting.insert(node);

            // Traverse through op inputs first
            if (node->op) {
                auto op_inputs = node->op->get_inputs();
                for (auto* inp : op_inputs) {
                    if (inp && inp->requires_grad) {
                        dfs(inp);
                    }
                }
            }

            // Also traverse tensor inputs
            for (auto* inp : node->inputs) {
                dfs(inp);
            }

            visiting.erase(node);
            visited.insert(node);
            backward_order_.push_back(node);
        };

        dfs(loss);
        std::reverse(backward_order_.begin(), backward_order_.end());

        if (debug_mode_) {
            std::cout << "[BACKWARD] Topological order: " << backward_order_.size() << " tensors\n";
            for (size_t i = 0; i < backward_order_.size(); ++i) {
                auto* t = backward_order_[i];
                std::cout << "  [" << i << "] shape=" << (t ? t->shape.size() : 0) 
                          << " is_leaf=" << (t ? t->is_leaf : 0) << "\n";
            }
        }

        // STEP 2: Initialize loss gradient (seed: d(loss)/d(loss) = 1)
        if (!loss->grad) {
            loss->grad = std::make_shared<Tensor>(loss->shape);
        }
        loss->grad->fill(1.0f);

        // STEP 3: Ensure all tensors in backward order have grad buffers
        for (auto* t : backward_order_) {
            if (!t || !t->requires_grad) continue;
            if (!t->grad) {
                t->grad = std::make_shared<Tensor>(t->shape);
                t->grad->zero_();
            }
        }

        // STEP 4: Process in reverse topological order (from loss towards inputs)
        for (auto* current : backward_order_) {
            if (!current || !current->requires_grad) continue;
            if (!current->grad || !current->grad->data) continue;

            if (current->op) {
                current->op->backward(*current);
                
                if (debug_mode_ && current->has_nan()) {
                    std::cerr << "[WARN] NaN detected after backward for op at shape=" 
                              << current->shape.size() << "\n";
                }
            }
        }

        // STEP 5: CRITICAL FIX - Visit ALL registered parameters and ensure they have gradients
        // Parameters that are not in the direct graph path (e.g., weights used in forward but
        // not connected via requires_grad chain) need explicit gradient accumulation
        for (auto* param : parameters_) {
            if (!param || !param->requires_grad) continue;
            
            // If param was already visited in the graph traversal, its grad is already computed
            if (visited.count(param)) {
                continue;
            }
            
            // If param was NOT visited, it means it's not connected in the graph
            // This is a bug - but we'll create a grad buffer anyway so optimizers don't crash
            if (!param->grad) {
                param->grad = std::make_shared<Tensor>(param->shape);
                param->grad->zero_();
            }
            
            if (debug_mode_) {
                std::cout << "[WARN] Parameter shape=" << param->shape.size() 
                          << " was NOT visited in graph traversal!\n";
            }
        }

        if (debug_mode_) {
            std::cout << "\n=== GRADIENT FLOW REPORT ===" << std::endl;
            for (auto* param : parameters_) {
                if (!param || !param->grad || !param->grad->data) {
                    std::cout << "[PARAM] shape=" << (param ? param->shape.size() : 0) 
                              << " grad=MISSING\n";
                    continue;
                }
                
                float max_grad = 0.0f;
                float mean_grad = 0.0f;
                size_t n = param->grad->shape.size();
                
                for (size_t i = 0; i < n; ++i) {
                    float g = std::abs(param->grad->data[param->grad->offset + i]);
                    max_grad = std::max(max_grad, g);
                    mean_grad += g;
                }
                mean_grad /= n;
                
                std::cout << "[PARAM] shape=" << param->shape.size() 
                          << " max_grad=" << max_grad 
                          << " mean_grad=" << mean_grad << "\n";
            }
            std::cout << "============================\n";
        }
    }

    // =========================================================================
    // ZERO GRAD
    // =========================================================================

    void zero_grad() {
        if (debug_mode_) {
            std::cout << "[ZERO_GRAD] Clearing gradients for " << parameters_.size() << " parameters\n";
        }

        if (accumulate_gradients_) {
            current_step_++;
            if (current_step_ >= accumulation_steps_) {
                // Average accumulated gradients
                for (auto* t : parameters_) {
                    if (t && t->grad && t->grad->data) {
                        float* g = t->grad->data.get() + t->grad->offset;
                        size_t n = t->grad->shape.size();
                        for (size_t i = 0; i < n; ++i) {
                            g[i] /= static_cast<float>(accumulation_steps_);
                        }
                    }
                }
                current_step_ = 0;
            }
        } else {
            // Zero out gradients for all registered parameters
            for (auto* t : parameters_) {
                if (t && t->grad && t->grad->data) {
                    t->grad->zero_();
                }
            }
        }
    }

    void step() {
        zero_grad();
    }

    // =========================================================================
    // UTILITIES
    // =========================================================================

    void check_nan(Tensor* t, const char* context) {
        if (!t || !t->data) return;
        for (size_t i = 0; i < t->shape.size(); ++i) {
            float v = t->data[t->offset + i];
            if (std::isnan(v) || std::isinf(v)) {
                std::cerr << "[NaN] Detected in " << context << " at index " << i << " value=" << v << "\n";
                throw std::runtime_error("NaN/Inf detected");
            }
        }
    }

    void collect_garbage() {
        int collected = 0;
        for (auto* t : leaf_tensors_) {
            if (t && t->data && !t->requires_grad && t->grad) {
                t->grad.reset();
                collected++;
            }
        }
        if (debug_mode_) {
            std::cout << "[GC] Collected " << collected << " tensors\n";
        }
    }

    void clear() {
        leaf_tensors_.clear();
        backward_order_.clear();
    }

    void clear_all() {
        clear();
        parameters_.clear();
    }

    const std::vector<Tensor*>& get_backward_order() const {
        return backward_order_;
    }

    const std::unordered_set<Tensor*>& get_leaf_tensors() const {
        return leaf_tensors_;
    }
    
    size_t parameter_count() const {
        return parameters_.size();
    }
};

// ============================================================================
// Thread-local context (global singleton per thread)
// ============================================================================

inline ExecutionContext& get_default_context() {
    static thread_local ExecutionContext default_context("thread_default");
    return default_context;
}

inline ExecutionContext& get_engine() {
    return get_default_context();
}

// ============================================================================
// Global API functions
// ============================================================================

inline void backward(Tensor& t) {
    get_engine().backward(&t);
}

inline void zero_grad() {
    get_engine().zero_grad();
}

inline void set_debug(bool d) {
    get_engine().set_debug(d);
}

inline void clip_grad(float max_norm) {
    get_engine().clip_grad(max_norm);
}

// Parameter registry access
inline void register_parameter(Tensor* param) {
    get_engine().register_parameter(param);
}

inline const std::vector<Tensor*>& get_parameters() {
    return get_engine().get_parameters();
}

inline void clear_parameters() {
    get_engine().clear_parameters();
}

inline size_t parameter_count() {
    return get_engine().parameter_count();
}

// ============================================================================
// Context factory
// ============================================================================

inline std::unique_ptr<ExecutionContext> create_context(const std::string& name = "model") {
    return std::make_unique<ExecutionContext>(name);
}

} // namespace aresml
