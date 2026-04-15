#pragma once

#include "../core/Tensor.hpp"
#include "../core/Autograd.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <iostream>

namespace aresml {

enum class OpType {
    UNKNOWN,
    LINEAR,
    MATMUL,
    ADD,
    RELU,
    SILU,
    SOFTMAX,
    LAYER_NORM,
    RMS_NORM,
    ATTENTION,
    EMBDDING,
    DROPOUT,
    FUSED_LINEAR_RELU,
    FUSED_LINEAR_ADD_RELU,
    FUSED_GEMM_BIAS,
};

struct GraphNode {
    OpType type;
    std::vector<Tensor*> inputs;
    Tensor* output;
    std::string name;
    
    GraphNode(OpType t, const std::vector<Tensor*>& in, Tensor* out, const std::string& n = "")
        : type(t), inputs(in), output(out), name(n) {}
};

class ComputationGraph {
public:
    std::vector<GraphNode> nodes;
    std::unordered_map<Tensor*, size_t> tensor_to_node;
    
    void add_node(OpType type, const std::vector<Tensor*>& inputs, Tensor* output, const std::string& name = "") {
        GraphNode node(type, inputs, output, name);
        tensor_to_node[output] = nodes.size();
        nodes.push_back(node);
    }
    
    bool has_node(const Tensor* t) const {
        return tensor_to_node.count(const_cast<Tensor*>(t));
    }
    
    size_t node_count() const { return nodes.size(); }
    
    std::string to_string() const {
        std::ostringstream ss;
        ss << "Graph(" << nodes.size() << " nodes):\n";
        for (size_t i = 0; i < nodes.size(); ++i) {
            ss << "  " << i << ": " << op_type_name(nodes[i].type);
            if (!nodes[i].name.empty()) ss << " [" << nodes[i].name << "]";
            ss << "\n";
        }
        return ss.str();
    }
    
private:
    static std::string op_type_name(OpType t) {
        switch(t) {
            case OpType::LINEAR: return "Linear";
            case OpType::MATMUL: return "Matmul";
            case OpType::ADD: return "Add";
            case OpType::RELU: return "ReLU";
            case OpType::SILU: return "SiLU";
            case OpType::SOFTMAX: return "Softmax";
            case OpType::ATTENTION: return "Attention";
            case OpType::FUSED_LINEAR_RELU: return "FusedLinearReLU";
            case OpType::FUSED_GEMM_BIAS: return "FusedGemmBias";
            default: return "Unknown";
        }
    }
};

class GraphRecorder {
public:
    static bool capture_enabled;
    static ComputationGraph* current_graph;
    static bool debug_mode;
    
    static void enable(bool enable) {
        capture_enabled = enable;
        if (enable) {
            std::cout << "[GRAPH] Recording enabled\n";
        }
    }
    
    static void enable_debug(bool enable) {
        debug_mode = enable;
    }
    
    static void begin_capture() {
        if (capture_enabled) {
            current_graph = new ComputationGraph();
            if (debug_mode) std::cout << "[GRAPH] Capture started\n";
        }
    }
    
    static ComputationGraph* end_capture() {
        if (capture_enabled && current_graph) {
            if (debug_mode) {
                std::cout << "[GRAPH] Capture ended: " << current_graph->node_count() << " nodes\n";
                std::cout << current_graph->to_string();
            }
            return current_graph;
        }
        return nullptr;
    }
    
    static void record(OpType type, const std::vector<Tensor*>& inputs, Tensor* output, const std::string& name = "") {
        if (capture_enabled && current_graph) {
            current_graph->add_node(type, inputs, output, name);
            if (debug_mode) {
                std::cout << "[GRAPH] Recorded: " << op_type_name(type);
                if (!name.empty()) std::cout << " [" << name << "]";
                std::cout << "\n";
            }
        }
    }
    
    static void record_linear(Tensor* input, Tensor* weight, Tensor* output, const std::string& name = "") {
        record(OpType::LINEAR, {input, weight}, output, name);
    }
    
    static void record_matmul(Tensor* a, Tensor* b, Tensor* output, const std::string& name = "") {
        record(OpType::MATMUL, {a, b}, output, name);
    }
    
    static void record_relu(Tensor* input, Tensor* output, const std::string& name = "") {
        record(OpType::RELU, {input}, output, name);
    }
    
    static void record_silu(Tensor* input, Tensor* output, const std::string& name = "") {
        record(OpType::SILU, {input}, output, name);
    }
    
    static void record_attention(Tensor* q, Tensor* k, Tensor* v, Tensor* output, const std::string& name = "") {
        record(OpType::ATTENTION, {q, k, v}, output, name);
    }
    
private:
    static const char* op_type_name(OpType t) {
        switch(t) {
            case OpType::LINEAR: return "Linear";
            case OpType::MATMUL: return "Matmul";
            case OpType::RELU: return "ReLU";
            case OpType::ATTENTION: return "Attention";
            default: return "Op";
        }
    }
};

bool GraphRecorder::capture_enabled = false;
ComputationGraph* GraphRecorder::current_graph = nullptr;
bool GraphRecorder::debug_mode = false;

class GraphOptimizer {
public:
    static void optimize(ComputationGraph& graph) {
        fuse_operations(graph);
        if (GraphRecorder::debug_mode) {
            std::cout << "[OPTIMIZER] After fusion: " << graph.node_count() << " nodes\n";
        }
    }
    
private:
    static void fuse_operations(ComputationGraph& graph) {
        std::vector<size_t> to_remove;
        
        for (size_t i = 0; i + 1 < graph.nodes.size(); ++i) {
            GraphNode& a = graph.nodes[i];
            GraphNode& b = graph.nodes[i + 1];
            
            if (can_fuse(a.type, b.type)) {
                a.type = OpType::FUSED_LINEAR_RELU;
                to_remove.push_back(i + 1);
                
                if (GraphRecorder::debug_mode) {
                    std::cout << "[OPTIMIZER] Fused nodes " << i << " and " << i+1 << "\n";
                }
            }
        }
        
        std::reverse(to_remove.begin(), to_remove.end());
        for (size_t idx : to_remove) {
            graph.nodes.erase(graph.nodes.begin() + idx);
        }
    }
    
    static bool can_fuse(OpType a, OpType b) {
        return (a == OpType::LINEAR || a == OpType::MATMUL) && (b == OpType::RELU || b == OpType::SILU);
    }
};

class FusedExecutor {
public:
    static void execute_fused_linear_relu(const Tensor& input, const Tensor& weight, Tensor& output) {
        if (!input.data || !weight.data || !output.data) return;
        
        const float* in = input.data.get() + input.offset;
        const float* w = weight.data.get() + weight.offset;
        float* out = output.data.get() + output.offset;
        
        size_t M = input.shape[0];
        size_t K = input.shape[1];
        size_t N = weight.shape[1];
        
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += in[i * K + k] * w[k * N + j];
                }
                out[i * N + j] = sum > 0.0f ? sum : 0.0f;
            }
        }
    }
    
    static void execute_fused_gemm_bias(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output) {
        if (!input.data || !weight.data || !output.data) return;
        if (!bias.data) {
            execute_fused_linear_relu(input, weight, output);
            return;
        }
        
        const float* in = input.data.get() + input.offset;
        const float* w = weight.data.get() + weight.offset;
        const float* b = bias.data.get() + bias.offset;
        float* out = output.data.get() + output.offset;
        
        size_t M = input.shape[0];
        size_t K = input.shape[1];
        size_t N = weight.shape[1];
        
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = b[j];
                for (size_t k = 0; k < K; ++k) {
                    sum += in[i * K + k] * w[k * N + j];
                }
                out[i * N + j] = sum;
            }
        }
    }
};

class GraphCompiler {
public:
    static void enable_compiler_mode(bool enable) {
        GraphRecorder::enable(enable);
        if (enable) {
            std::cout << "[COMPILER] Graph compiler mode enabled\n";
        } else {
            std::cout << "[COMPILER] Eager mode enabled\n";
        }
    }
    
    static void compile_and_execute(ComputationGraph& graph) {
        GraphOptimizer::optimize(graph);
        
        if (GraphRecorder::debug_mode) {
            std::cout << "[COMPILER] Executing optimized graph with " << graph.node_count() << " fused ops\n";
        }
    }
};

inline void set_compiler_mode(bool enable) {
    GraphCompiler::enable_compiler_mode(enable);
}

inline void set_compiler_debug(bool enable) {
    GraphRecorder::enable_debug(enable);
}

}