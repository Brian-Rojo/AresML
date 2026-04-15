#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include "../ir_v2/IRGraphV2.hpp"
#include "../scheduler/SchedulerV2.hpp"
#include "../memory/MemoryPlan.hpp"
#include "../memory/CachePlanner.hpp"
#include "../prefetch/PrefetchEngine.hpp"
#include "../simd/SimdKernelsV2.hpp"

namespace aresml {

class FusedKernelExecutor {
public:
    FusedKernelExecutor() 
        : num_threads_(4)
        , prefetch_enabled_(true)
        , simd_enabled_(true) {}
    
    explicit FusedKernelExecutor(size_t threads) 
        : num_threads_(threads)
        , prefetch_enabled_(true)
        , simd_enabled_(true) {}
    
    void execute(IRGraphV2& graph, const ExecutionPlan& plan) {
        execute_internal(graph, plan);
    }
    
    void execute_fused(IRGraphV2& graph, const ExecutionPlan& plan) {
        if (simd_enabled_) {
            execute_with_simd(graph, plan);
        } else {
            execute_internal(graph, plan);
        }
    }
    
    void set_num_threads(size_t n) { num_threads_ = n; }
    void set_prefetch_enabled(bool enabled) { prefetch_enabled_ = enabled; }
    void set_simd_enabled(bool enabled) { simd_enabled_ = enabled; }
    
    size_t get_num_threads() const { return num_threads_; }
    bool is_prefetch_enabled() const { return prefetch_enabled_; }
    bool is_simd_enabled() const { return simd_enabled_; }
    
private:
    size_t num_threads_;
    bool prefetch_enabled_;
    bool simd_enabled_;
    PrefetchEngine prefetch_engine_;
    
    void execute_internal(IRGraphV2& graph, const ExecutionPlan& plan) {
        for (const auto& task : plan.tasks) {
            if (!task.node) continue;
            
            if (task.stage == ExecutionStage::PREFETCH && prefetch_enabled_) {
                prefetch_inputs(task.node);
            }
            
            if (task.stage == ExecutionStage::COMPUTE) {
                execute_node(task.node);
            }
        }
    }
    
    void execute_with_simd(IRGraphV2& graph, const ExecutionPlan& plan) {
        for (const auto& task : plan.tasks) {
            if (!task.node) continue;
            
            if (task.is_fused) {
                execute_fused_node(task.node);
            } else {
                execute_node_simd(task.node);
            }
        }
    }
    
    void execute_node(IRNodeV2* node) {
        if (!node) return;
        
        switch(node->op) {
            case IROpV2::ADD:
                execute_add(node);
                break;
            case IROpV2::MUL:
                execute_mul(node);
                break;
            case IROpV2::MATMUL:
                execute_matmul(node);
                break;
            case IROpV2::RELU:
                execute_relu(node);
                break;
            case IROpV2::GELU:
                execute_gelu(node);
                break;
            case IROpV2::FUSED_GEMM_BIAS_RELU:
                execute_fused_gemm_bias_relu(node);
                break;
            case IROpV2::FUSED_MLP_BLOCK:
                execute_fused_mlp_block(node);
                break;
            case IROpV2::FUSED_ATTENTION_BLOCK:
                execute_fused_attention_block(node);
                break;
            default:
                break;
        }
    }
    
    void execute_node_simd(IRNodeV2* node) {
        if (!node) return;
        
        using namespace simd;
        
        switch(node->op) {
            case IROpV2::ADD:
                if (node->operands.size() >= 2) {
                    // execute_add_simd(node);
                }
                break;
            case IROpV2::MUL:
                // execute_mul_simd(node);
                break;
            case IROpV2::RELU:
                // execute_relu_simd(node);
                break;
            case IROpV2::GELU:
                // execute_gelu_simd(node);
                break;
            case IROpV2::MATMUL:
                execute_matmul_blocked(node);
                break;
            default:
                execute_node(node);
        }
    }
    
    void execute_fused_node(IRNodeV2* node) {
        if (!node) return;
        
        switch(node->op) {
            case IROpV2::FUSED_GEMM_BIAS_RELU:
                execute_fused_gemm_bias_relu(node);
                break;
            case IROpV2::FUSED_MLP_BLOCK:
                execute_fused_mlp_block(node);
                break;
            case IROpV2::FUSED_ATTENTION_BLOCK:
                execute_fused_attention_block(node);
                break;
            case IROpV2::FUSED_ELEMENTWISE_BLOCK:
                execute_fused_elementwise_block(node);
                break;
            default:
                execute_node(node);
        }
    }
    
    void execute_add(IRNodeV2* node) {}
    void execute_mul(IRNodeV2* node) {}
    
    void execute_matmul(IRNodeV2* node) {
        if (node->shape.dims.size() < 2) return;
        
        size_t m = node->shape.dims[0];
        size_t n = node->shape.dims[1];
        size_t k = node->shape.dims.size() > 2 ? node->shape.dims[2] : m;
    }
    
    void execute_matmul_blocked(IRNodeV2* node) {
        if (node->shape.dims.size() < 2) return;
        
        size_t m = node->shape.dims[0];
        size_t n = node->shape.dims[1];
        size_t k = node->shape.dims.size() > 2 ? node->shape.dims[2] : m;
        
        CacheBlock block = CachePlanner::compute_optimal_block(m, n, k);
        
        // simd::matmul_blocked(out, a, b, m, n, k, block.block_m, block.block_n, block.block_k);
    }
    
    void execute_relu(IRNodeV2* node) {}
    void execute_gelu(IRNodeV2* node) {}
    
    void execute_fused_gemm_bias_relu(IRNodeV2* node) {
        if (node->shape.dims.size() < 2) return;
        
        size_t m = node->shape.dims[0];
        size_t n = node->shape.dims[1];
        size_t k = node->shape.dims.size() > 2 ? node->shape.dims[2] : m;
        
        CacheBlock block = CachePlanner::compute_optimal_block(m, n, k);
        
        // Fused: matmul + bias + relu in single kernel
    }
    
    void execute_fused_mlp_block(IRNodeV2* node) {
        // Linear -> Relu -> Linear -> Relu fusion
    }
    
    void execute_fused_attention_block(IRNodeV2* node) {
        // QKV -> Matmul -> Softmax fusion
    }
    
    void execute_fused_elementwise_block(IRNodeV2* node) {
        // Add -> Mul fusion
    }
    
    void prefetch_inputs(IRNodeV2* node) {
        if (!node || !prefetch_enabled_) return;
        
        for (const auto& use : node->operands) {
            if (use.value) {
                // prefetch_engine_.prefetch_elementwise(...)
            }
        }
    }
};

inline void execute_ir_graph(IRGraphV2& graph) {
    SchedulerV2 scheduler;
    ExecutionPlan plan = scheduler.create_plan(graph);
    
    FusedKernelExecutor executor;
    executor.execute(graph, plan);
}

inline void execute_ir_graph_parallel(IRGraphV2& graph, size_t threads = 4) {
    SchedulerV2 scheduler(ScheduleStrategy::PARALLEL, threads);
    ExecutionPlan plan = scheduler.create_plan(graph);
    
    FusedKernelExecutor executor(threads);
    executor.execute(graph, plan);
}

inline void execute_ir_graph_fused(IRGraphV2& graph) {
    SchedulerV2 scheduler(ScheduleStrategy::FUSED);
    ExecutionPlan plan = scheduler.create_plan(graph);
    
    FusedKernelExecutor executor;
    executor.execute_fused(graph, plan);
}

}
