#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include "../ir_v3/IRNodeV3.hpp"
#include "../kernel_specialization/KernelDB.hpp"

namespace aresml {

struct KernelTask {
    size_t task_id;
    std::string kernel_name;
    IROpV3 op;
    KernelSpec spec;
    std::vector<size_t> input_ptrs;
    std::vector<size_t> output_ptrs;
    size_t offset_in_plan;
    bool is_fused;
    size_t estimated_cycles;
    
    KernelTask() : task_id(0), op(IROpV3::UNKNOWN), offset_in_plan(0), is_fused(false), estimated_cycles(0) {}
    
    KernelTask(size_t id, const std::string& name, IROpV3 o, const KernelSpec& s)
        : task_id(id), kernel_name(name), op(o), spec(s), offset_in_plan(0), is_fused(false), estimated_cycles(0) {}
};

class ExecutionPlan {
public:
    std::vector<KernelTask> tasks;
    std::vector<size_t> input_buffer_offsets;
    std::vector<size_t> output_buffer_offsets;
    std::vector<size_t> temp_buffer_offsets;
    
    size_t total_memory_needed;
    size_t peak_memory;
    size_t estimated_cycles;
    bool is_static;
    std::string graph_hash;
    
    ExecutionPlan() : total_memory_needed(0), peak_memory(0), estimated_cycles(0), is_static(true) {}
    
    ExecutionPlan(const ExecutionPlan& other)
        : tasks(other.tasks)
        , input_buffer_offsets(other.input_buffer_offsets)
        , output_buffer_offsets(other.output_buffer_offsets)
        , temp_buffer_offsets(other.temp_buffer_offsets)
        , total_memory_needed(other.total_memory_needed)
        , peak_memory(other.peak_memory)
        , estimated_cycles(other.estimated_cycles)
        , is_static(other.is_static)
        , graph_hash(other.graph_hash) {}
    
    ExecutionPlan& operator=(const ExecutionPlan& other) {
        if (this != &other) {
            tasks = other.tasks;
            input_buffer_offsets = other.input_buffer_offsets;
            output_buffer_offsets = other.output_buffer_offsets;
            temp_buffer_offsets = other.temp_buffer_offsets;
            total_memory_needed = other.total_memory_needed;
            peak_memory = other.peak_memory;
            estimated_cycles = other.estimated_cycles;
            is_static = other.is_static;
            graph_hash = other.graph_hash;
        }
        return *this;
    }
    
    void add_task(const KernelTask& task) {
        tasks.push_back(task);
    }
    
    void compute_offsets() {
        size_t offset = 0;
        
        for (auto& off : input_buffer_offsets) {
            off = offset;
            offset += 64 * 1024;
        }
        
        for (auto& off : output_buffer_offsets) {
            off = offset;
            offset += 64 * 1024;
        }
        
        for (auto& off : temp_buffer_offsets) {
            off = offset;
            offset += 64 * 1024;
        }
        
        total_memory_needed = offset;
    }
    
    size_t task_count() const { return tasks.size(); }
    
    std::string to_string() const {
        std::string s = "ExecutionPlan:\n";
        s += "  Tasks: " + std::to_string(tasks.size()) + "\n";
        s += "  Total memory: " + std::to_string(total_memory_needed) + " bytes\n";
        s += "  Peak memory: " + std::to_string(peak_memory) + " bytes\n";
        s += "  Estimated cycles: " + std::to_string(estimated_cycles) + "\n";
        s += "  Static: " + std::string(is_static ? "yes" : "no") + "\n";
        s += "  Graph hash: " + graph_hash + "\n";
        return s;
    }
    
    KernelTask* get_task(size_t index) {
        return index < tasks.size() ? &tasks[index] : nullptr;
    }
    
    const KernelTask* get_task(size_t index) const {
        return index < tasks.size() ? &tasks[index] : nullptr;
    }
};

}
