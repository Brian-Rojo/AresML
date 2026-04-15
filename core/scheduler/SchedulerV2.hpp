#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <functional>
#include "../ir_v2/IRGraphV2.hpp"
#include "../memory/MemoryPlan.hpp"
#include "../fusion/FusionGroups.hpp"

namespace aresml {

enum class ScheduleStrategy {
    SEQUENTIAL,
    PARALLEL,
    PIPELINE,
    FUSED,
    ADAPTIVE
};

enum class ExecutionStage {
    UNSCHEDULED,
    PREFETCH,
    COMPUTE,
    POSTPROCESS,
    COMPLETED
};

struct ExecutionTask {
    size_t task_id;
    IRNodeV2* node;
    ExecutionStage stage;
    size_t priority;
    size_t estimated_cycles;
    std::vector<size_t> dependencies;
    bool is_fused;
    bool is_parallel;
    
    ExecutionTask() 
        : task_id(0), node(nullptr), stage(ExecutionStage::UNSCHEDULED)
        , priority(0), estimated_cycles(0), is_fused(false), is_parallel(false) {}
    
    ExecutionTask(size_t id, IRNodeV2* n) 
        : task_id(id), node(n), stage(ExecutionStage::UNSCHEDULED)
        , priority(0), estimated_cycles(0), is_fused(false), is_parallel(false) {}
};

struct ExecutionPlan {
    std::vector<ExecutionTask> tasks;
    std::vector<std::vector<size_t>> parallel_groups;
    std::vector<size_t> execution_order;
    MemoryPlan memory_plan;
    ScheduleStrategy strategy;
    size_t total_cycles;
    size_t parallel_ops;
    float estimated_speedup;
    
    ExecutionPlan() 
        : strategy(ScheduleStrategy::SEQUENTIAL)
        , total_cycles(0), parallel_ops(0), estimated_speedup(1.0f) {}
    
    std::string to_string() const {
        std::string s = "ExecutionPlan:\n";
        s += "  Strategy: ";
        switch(strategy) {
            case ScheduleStrategy::SEQUENTIAL: s += "SEQUENTIAL\n"; break;
            case ScheduleStrategy::PARALLEL: s += "PARALLEL\n"; break;
            case ScheduleStrategy::PIPELINE: s += "PIPELINE\n"; break;
            case ScheduleStrategy::FUSED: s += "FUSED\n"; break;
            case ScheduleStrategy::ADAPTIVE: s += "ADAPTIVE\n"; break;
        }
        s += "  Tasks: " + std::to_string(tasks.size()) + "\n";
        s += "  Parallel groups: " + std::to_string(parallel_groups.size()) + "\n";
        s += "  Total cycles: " + std::to_string(total_cycles) + "\n";
        s += "  Estimated speedup: " + std::to_string(estimated_speedup) + "x\n";
        return s;
    }
};

class SchedulerV2 {
public:
    SchedulerV2() : strategy_(ScheduleStrategy::SEQUENTIAL), num_threads_(4) {}
    
    explicit SchedulerV2(ScheduleStrategy strategy) 
        : strategy_(strategy), num_threads_(4) {}
    
    SchedulerV2(ScheduleStrategy strategy, size_t threads) 
        : strategy_(strategy), num_threads_(threads) {}
    
    ExecutionPlan create_plan(IRGraphV2& graph) {
        ExecutionPlan plan;
        plan.strategy = strategy_;
        
        switch(strategy_) {
            case ScheduleStrategy::SEQUENTIAL:
                plan = create_sequential_plan(graph);
                break;
            case ScheduleStrategy::PARALLEL:
                plan = create_parallel_plan(graph);
                break;
            case ScheduleStrategy::FUSED:
                plan = create_fused_plan(graph);
                break;
            case ScheduleStrategy::ADAPTIVE:
                plan = create_adaptive_plan(graph);
                break;
            default:
                plan = create_sequential_plan(graph);
        }
        
        return plan;
    }
    
    void set_strategy(ScheduleStrategy strategy) { strategy_ = strategy; }
    void set_num_threads(size_t n) { num_threads_ = n; }
    
    ScheduleStrategy get_strategy() const { return strategy_; }
    size_t get_num_threads() const { return num_threads_; }
    
private:
    ScheduleStrategy strategy_;
    size_t num_threads_;
    
    ExecutionPlan create_sequential_plan(IRGraphV2& graph) {
        ExecutionPlan plan;
        plan.strategy = ScheduleStrategy::SEQUENTIAL;
        
        for (size_t i = 0; i < graph.nodes.size(); ++i) {
            ExecutionTask task(i, graph.nodes[i].get());
            task.stage = ExecutionStage::COMPUTE;
            task.priority = graph.nodes.size() - i;
            task.estimated_cycles = estimate_cycles(task.node);
            plan.tasks.push_back(task);
            plan.execution_order.push_back(i);
        }
        
        plan.total_cycles = compute_total_cycles(plan);
        plan.estimated_speedup = 1.0f;
        
        return plan;
    }
    
    ExecutionPlan create_parallel_plan(IRGraphV2& graph) {
        ExecutionPlan plan;
        plan.strategy = ScheduleStrategy::PARALLEL;
        
        std::vector<ExecutionTask> elementwise_tasks;
        std::vector<ExecutionTask> matmul_tasks;
        std::vector<ExecutionTask> other_tasks;
        
        for (size_t i = 0; i < graph.nodes.size(); ++i) {
            ExecutionTask task(i, graph.nodes[i].get());
            task.stage = ExecutionStage::COMPUTE;
            task.estimated_cycles = estimate_cycles(task.node);
            
            if (task.node->op == IROpV2::ADD || 
                task.node->op == IROpV2::MUL ||
                task.node->op == IROpV2::RELU) {
                elementwise_tasks.push_back(task);
            } else if (task.node->op == IROpV2::MATMUL) {
                matmul_tasks.push_back(task);
            } else {
                other_tasks.push_back(task);
            }
        }
        
        if (!elementwise_tasks.empty()) {
            plan.parallel_groups.push_back({});
            for (auto& t : elementwise_tasks) {
                t.is_parallel = true;
                plan.parallel_groups.back().push_back(t.task_id);
            }
        }
        
        size_t task_id = 0;
        for (auto& t : other_tasks) {
            t.task_id = task_id++;
            plan.tasks.push_back(t);
        }
        for (auto& t : matmul_tasks) {
            t.task_id = task_id++;
            plan.tasks.push_back(t);
        }
        for (auto& t : elementwise_tasks) {
            t.task_id = task_id++;
            plan.tasks.push_back(t);
        }
        
        plan.total_cycles = compute_total_cycles(plan);
        plan.parallel_ops = elementwise_tasks.size();
        plan.estimated_speedup = 1.0f + static_cast<float>(plan.parallel_ops) / static_cast<float>(num_threads_);
        
        return plan;
    }
    
    ExecutionPlan create_fused_plan(IRGraphV2& graph) {
        ExecutionPlan plan;
        plan.strategy = ScheduleStrategy::FUSED;
        
        for (size_t i = 0; i < graph.nodes.size(); ++i) {
            auto* node = graph.nodes[i].get();
            ExecutionTask task(i, node);
            
            if (node->is_fused()) {
                task.is_fused = true;
                task.stage = ExecutionStage::COMPUTE;
                task.estimated_cycles = estimate_cycles(node) / 2;
            } else {
                task.stage = ExecutionStage::COMPUTE;
                task.estimated_cycles = estimate_cycles(node);
            }
            
            plan.tasks.push_back(task);
            plan.execution_order.push_back(i);
        }
        
        plan.total_cycles = compute_total_cycles(plan);
        plan.estimated_speedup = 2.5f;
        
        return plan;
    }
    
    ExecutionPlan create_adaptive_plan(IRGraphV2& graph) {
        ExecutionPlan plan;
        plan.strategy = ScheduleStrategy::ADAPTIVE;
        
        size_t fusion_count = 0;
        for (auto& node : graph.nodes) {
            if (node->is_fused()) fusion_count++;
        }
        
        if (fusion_count > graph.nodes.size() / 2) {
            return create_fused_plan(graph);
        } else if (graph.nodes.size() > 10) {
            return create_parallel_plan(graph);
        }
        
        return create_sequential_plan(graph);
    }
    
    size_t estimate_cycles(IRNodeV2* node) const {
        if (!node) return 1000;
        
        switch(node->op) {
            case IROpV2::MATMUL: {
                size_t m = node->shape.dims[0];
                size_t n = node->shape.dims[1];
                size_t k = node->shape.dims.size() > 2 ? node->shape.dims[2] : m;
                return m * n * k;
            }
            case IROpV2::ADD:
            case IROpV2::MUL:
                return node->shape.numel() * 2;
            case IROpV2::RELU:
            case IROpV2::GELU:
                return node->shape.numel() * 3;
            case IROpV2::SOFTMAX:
                return node->shape.numel() * 10;
            default:
                return node->shape.numel();
        }
    }
    
    size_t compute_total_cycles(const ExecutionPlan& plan) const {
        size_t total = 0;
        for (const auto& task : plan.tasks) {
            total += task.estimated_cycles;
        }
        return total;
    }
};

inline ExecutionPlan schedule_ir_graph(IRGraphV2& graph) {
    SchedulerV2 scheduler;
    return scheduler.create_plan(graph);
}

inline ExecutionPlan schedule_ir_graph(IRGraphV2& graph, ScheduleStrategy strategy) {
    SchedulerV2 scheduler(strategy);
    return scheduler.create_plan(graph);
}

}
