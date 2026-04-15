#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../ir_v2/IRGraphV2.hpp"
#include "../ir_v3/IRGraphV3.hpp"
#include "GraphLoweringV3.hpp"
#include "PassManagerV3.hpp"
#include "../kernel_specialization/KernelDB.hpp"
#include "../execution_v3/ExecutionPlan.hpp"
#include "../runtime_v3/PlanExecutor.hpp"

namespace aresml {

struct CompilationResult {
    std::string graph_hash;
    std::unique_ptr<IRGraphV3> v3_graph;
    std::unique_ptr<ExecutionPlan> plan;
    size_t compilation_time_ms;
    bool from_cache;
    size_t estimated_speedup;
    
    CompilationResult() : compilation_time_ms(0), from_cache(false), estimated_speedup(1.0f) {}
};

class CompilationCache {
public:
    static constexpr size_t MAX_CACHE_SIZE = 128;
    
    struct CachedPlan {
        std::string graph_hash;
        std::unique_ptr<ExecutionPlan> plan;
        size_t estimated_speedup;
        size_t access_count;
        
        CachedPlan() : estimated_speedup(1.0f), access_count(0) {}
    };
    
    void insert(const std::string& hash, const CompilationResult& result) {
        if (cache_.size() >= MAX_CACHE_SIZE) {
            cache_.erase(cache_.begin());
        }
        auto entry = std::make_unique<CachedPlan>();
        entry->graph_hash = result.graph_hash;
        entry->estimated_speedup = result.estimated_speedup;
        entry->plan = nullptr;
        if (result.plan) {
            entry->plan = std::make_unique<ExecutionPlan>();
            *entry->plan = *result.plan;
        }
        cache_[hash] = std::move(entry);
    }
    
    bool lookup(const std::string& hash, std::unique_ptr<ExecutionPlan>& plan, size_t& speedup) const {
        auto it = cache_.find(hash);
        if (it != cache_.end()) {
            if (it->second->plan) {
                plan = std::move(it->second->plan);
            }
            speedup = it->second->estimated_speedup;
            it->second->access_count++;
            return true;
        }
        return false;
    }
    
    bool lookup(const std::string& hash) const {
        return cache_.find(hash) != cache_.end();
    }
    
    void clear() { cache_.clear(); }
    size_t size() const { return cache_.size(); }
    
private:
    std::unordered_map<std::string, std::unique_ptr<CachedPlan>> cache_;
};

class GraphCompilerV3 {
public:
    GraphCompilerV3() : enable_cache_(true), enable_autotune_(false) {
        cache_ = std::make_unique<CompilationCache>();
    }
    
    explicit GraphCompilerV3(bool enable_cache, bool enable_autotune = false)
        : enable_cache_(enable_cache), enable_autotune_(enable_autotune) {
        cache_ = std::make_unique<CompilationCache>();
    }
    
    CompilationResult compile(IRGraphV2& v2_graph) {
        CompilationResult result;
        
        auto hash = compute_hash(v2_graph);
        result.graph_hash = hash;
        
        if (enable_cache_) {
            std::unique_ptr<ExecutionPlan> cached_plan;
            size_t speedup;
            if (cache_->lookup(hash, cached_plan, speedup)) {
                result.from_cache = true;
                result.plan = std::move(cached_plan);
                result.estimated_speedup = speedup;
                return result;
            }
        }
        
        auto lowering_result = lowering_.lower(v2_graph);
        result.v3_graph = std::move(lowering_result.graph);
        
        pass_manager_.run_until_fixed_point(*result.v3_graph);
        
        result.plan = create_execution_plan(*result.v3_graph);
        
        result.estimated_speedup = estimate_speedup(*result.v3_graph);
        
        if (enable_cache_) {
            cache_->insert(hash, result);
        }
        
        return result;
    }
    
    void set_cache_enabled(bool enabled) { enable_cache_ = enabled; }
    void set_autotune_enabled(bool enabled) { enable_autotune_ = enabled; }
    
    const CompilationCache& get_cache() const { return *cache_; }
    
    void clear_cache() { cache_->clear(); }
    
private:
    std::unique_ptr<CompilationCache> cache_;
    GraphLoweringV3 lowering_;
    PassManagerV3 pass_manager_;
    bool enable_cache_;
    bool enable_autotune_;
    
    std::string compute_hash(IRGraphV2& graph) const {
        std::string hash = std::to_string(graph.node_count());
        for (auto& node : graph.nodes) {
            hash += "|" + std::to_string(static_cast<int>(node->op));
            for (auto d : node->shape.dims) {
                hash += "," + std::to_string(d);
            }
        }
        return hash;
    }
    
    std::unique_ptr<ExecutionPlan> create_execution_plan(IRGraphV3& v3_graph) {
        auto plan = std::make_unique<ExecutionPlan>();
        
        for (size_t i = 0; i < v3_graph.instructions.size(); ++i) {
            auto& inst = v3_graph.instructions[i];
            
            auto spec = KernelDB::get_kernel(inst->op, inst->shape);
            
            KernelTask task(i, spec.name, inst->op, spec);
            task.estimated_cycles = spec.estimated_cycles;
            task.is_fused = inst->is_fused;
            
            plan->add_task(task);
        }
        
        plan->graph_hash = v3_graph.compute_hash();
        plan->compute_offsets();
        
        return plan;
    }
    
    size_t estimate_speedup(const IRGraphV3& v3_graph) const {
        size_t original_ops = 0;
        size_t fused_ops = 0;
        
        for (auto& inst : v3_graph.instructions) {
            original_ops++;
            if (inst->is_fused) {
                fused_ops++;
            }
        }
        
        if (original_ops == 0) return 1;
        
        float fused_ratio = static_cast<float>(fused_ops) / static_cast<float>(original_ops);
        
        return static_cast<size_t>(1.0f + fused_ratio * 3.0f);
    }
};

inline CompilationResult compile_graph(IRGraphV2& graph) {
    GraphCompilerV3 compiler;
    return compiler.compile(graph);
}

}
