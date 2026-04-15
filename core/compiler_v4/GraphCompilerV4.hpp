#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include "../ir_v2/IRGraphV2.hpp"
#include "../compiler_v3/GraphCompilerV3.hpp"
#include "../execution_v3/ExecutionPlan.hpp"

namespace aresml {

class OptimizationPipeline;

struct StaticExecutionNode {
    size_t node_id;
    std::string kernel_name;
    std::vector<size_t> input_buffer_ids;
    std::vector<size_t> output_buffer_ids;
    int32_t dependency_mask;
    bool is_fused;
    bool is_inplace;
    size_t estimated_cycles;
    size_t offset_in_plan;
    
    StaticExecutionNode() 
        : node_id(0), dependency_mask(0), is_fused(false), is_inplace(false)
        , estimated_cycles(0), offset_in_plan(0) {}
};

class StaticExecutionPlan {
public:
    std::vector<StaticExecutionNode> nodes;
    std::vector<size_t> input_buffers;
    std::vector<size_t> output_buffers;
    std::vector<size_t> temp_buffers;
    
    size_t total_memory;
    size_t kernel_count;
    size_t fused_kernel_count;
    bool is_static;
    std::string graph_hash;
    
    StaticExecutionPlan() 
        : total_memory(0), kernel_count(0), fused_kernel_count(0), is_static(true) {}
    
    void add_node(const StaticExecutionNode& node) {
        nodes.push_back(node);
    }
    
    size_t size() const { return nodes.size(); }
    
    bool empty() const { return nodes.empty(); }
    
    const StaticExecutionNode* get_node(size_t i) const {
        return i < nodes.size() ? &nodes[i] : nullptr;
    }
    
    std::string to_string() const {
        std::string s = "StaticExecutionPlan:\n";
        s += "  Nodes: " + std::to_string(nodes.size()) + "\n";
        s += "  Kernels: " + std::to_string(kernel_count) + "\n";
        s += "  Fused: " + std::to_string(fused_kernel_count) + "\n";
        s += "  Total memory: " + std::to_string(total_memory) + " bytes\n";
        s += "  Static: " + std::string(is_static ? "yes" : "no") + "\n";
        return s;
    }
};

class GraphCompilerV4 {
public:
    GraphCompilerV4() : enable_optimizations_(true), enable_fusion_(true) {}
    
    std::unique_ptr<StaticExecutionPlan> compile(IRGraphV2& graph) {
        auto plan = std::make_unique<StaticExecutionPlan>();
        
        plan->graph_hash = compute_hash(graph);
        
        run_optimizations(graph);
        
        compute_node_order(graph, *plan);
        
        assign_buffers(*plan);
        
        fuse_kernels(*plan);
        
        compute_memory_layout(*plan);
        
        return plan;
    }
    
    void set_optimizations_enabled(bool enabled) { enable_optimizations_ = enabled; }
    void set_fusion_enabled(bool enabled) { enable_fusion_ = enabled; }
    
private:
    bool enable_optimizations_;
    bool enable_fusion_;
    
    std::string compute_hash(IRGraphV2& graph) const {
        std::string hash = "v4_";
        hash += std::to_string(graph.node_count());
        for (auto& node : graph.nodes) {
            hash += "_" + std::to_string(static_cast<int>(node->op));
            for (auto d : node->shape.dims) {
                hash += "x" + std::to_string(d);
            }
        }
        return hash;
    }
    
    void run_optimizations(IRGraphV2& graph) {
        if (!enable_optimizations_) return;
        
        std::vector<size_t> to_remove;
        
        for (size_t i = 0; i < graph.nodes.size(); ++i) {
            auto& node = graph.nodes[i];
            if (node->op == IROpV2::INPUT || node->op == IROpV2::PARAM) {
                continue;
            }
            if (node->shape.numel() < 4) {
                to_remove.push_back(i);
            }
        }
    }
    
    void compute_node_order(IRGraphV2& graph, StaticExecutionPlan& plan) {
        for (size_t i = 0; i < graph.nodes.size(); ++i) {
            auto& node = graph.nodes[i];
            
            if (node->op == IROpV2::INPUT || node->op == IROpV2::PARAM) {
                plan.input_buffers.push_back(i);
                continue;
            }
            
            StaticExecutionNode exec_node;
            exec_node.node_id = i;
            exec_node.kernel_name = get_kernel_name(node->op);
            exec_node.estimated_cycles = estimate_cycles(node.get());
            exec_node.offset_in_plan = plan.nodes.size();
            
            plan.nodes.push_back(exec_node);
            plan.kernel_count++;
        }
    }
    
    void assign_buffers(StaticExecutionPlan& plan) {
        size_t buffer_id = 0;
        
        for (auto& node : plan.nodes) {
            for (size_t i = 0; i < 2; ++i) {
                node.input_buffer_ids.push_back(buffer_id++);
            }
            node.output_buffer_ids.push_back(buffer_id++);
        }
        
        plan.total_memory = buffer_id * sizeof(float) * 1024;
    }
    
    void fuse_kernels(StaticExecutionPlan& plan) {
        for (size_t i = 0; i + 1 < plan.nodes.size(); ++i) {
            auto& a = plan.nodes[i];
            auto& b = plan.nodes[i + 1];
            
            if (can_fuse(a.kernel_name, b.kernel_name)) {
                a.is_fused = true;
                b.is_fused = true;
                plan.fused_kernel_count++;
            }
        }
    }
    
    void compute_memory_layout(StaticExecutionPlan& plan) {
        for (auto& node : plan.nodes) {
            if (node.is_inplace) {
                node.output_buffer_ids = node.input_buffer_ids;
            }
        }
    }
    
    std::string get_kernel_name(IROpV2 op) const {
        switch(op) {
            case IROpV2::MATMUL: return "gemm";
            case IROpV2::ADD: return "add";
            case IROpV2::MUL: return "mul";
            case IROpV2::RELU: return "relu";
            case IROpV2::GELU: return "gelu";
            case IROpV2::SOFTMAX: return "softmax";
            case IROpV2::FUSED_GEMM_BIAS_RELU: return "fused_gemm_bias_relu";
            default: return "unknown";
        }
    }
    
    size_t estimate_cycles(IRNodeV2* node) const {
        if (!node) return 1000;
        size_t n = node->shape.numel();
        switch(node->op) {
            case IROpV2::MATMUL: return n * 2;
            case IROpV2::ADD:
            case IROpV2::MUL: return n;
            case IROpV2::RELU: return n / 2;
            default: return n;
        }
    }
    
    bool can_fuse(const std::string& a, const std::string& b) const {
        if (!enable_fusion_) return false;
        
        if (a == "gemm" && b == "add") return true;
        if (a == "add" && b == "relu") return true;
        if (a == "add" && b == "add") return true;
        if (a == "mul" && b == "add") return true;
        
        return false;
    }
};

inline std::unique_ptr<StaticExecutionPlan> compile_static(IRGraphV2& graph) {
    GraphCompilerV4 compiler;
    return compiler.compile(graph);
}

}
