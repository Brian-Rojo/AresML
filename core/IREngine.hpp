#pragma once

#include "../core/Tensor.hpp"
#include "../core/ShapeEngine.hpp"
#include "../core/GraphCompiler.hpp"
#include <vector>
#include <unordered_map>
#include <memory>
#include <sstream>
#include <iostream>
#include <functional>

namespace aresml {

enum class IRType {
    UNKNOWN,
    INPUT,
    PARAM,
    MATMUL,
    ADD,
    BIAS,
    RELU,
    SILU,
    SOFTMAX,
    ATTENTION_QKV,
    FUSED_LINEAR_RELU,
    FUSED_GEMM_BIAS,
    FUSED_GEMM_BIAS_RELU,
    OUTPUT,
};

struct IRNode {
    IRType type;
    std::vector<IRNode*> inputs;
    std::vector<Tensor*> output_tensors;
    TensorShape output_shape;
    std::unordered_map<std::string, TensorShape> input_shapes;
    std::string name;
    
    IRNode(IRType t, const TensorShape& shape, const std::string& n = "")
        : type(t), output_shape(shape), name(n) {}
    
    bool is_fused() const {
        return type == IRType::FUSED_LINEAR_RELU ||
               type == IRType::FUSED_GEMM_BIAS ||
               type == IRType::FUSED_GEMM_BIAS_RELU;
    }
    
    std::string type_name() const {
        switch(type) {
            case IRType::INPUT: return "Input";
            case IRType::PARAM: return "Param";
            case IRType::MATMUL: return "Matmul";
            case IRType::ADD: return "Add";
            case IRType::BIAS: return "Bias";
            case IRType::RELU: return "ReLU";
            case IRType::SILU: return "SiLU";
            case IRType::SOFTMAX: return "Softmax";
            case IRType::ATTENTION_QKV: return "AttentionQKV";
            case IRType::FUSED_LINEAR_RELU: return "FusedLinearReLU";
            case IRType::FUSED_GEMM_BIAS: return "FusedGemmBias";
            case IRType::FUSED_GEMM_BIAS_RELU: return "FusedGemmBiasReLU";
            default: return "Unknown";
        }
    }
};

class IRGraph {
public:
    std::vector<std::unique_ptr<IRNode>> nodes;
    std::unordered_map<std::string, IRNode*> node_map;
    
    IRNode* add_node(IRType type, const TensorShape& shape, const std::string& name = "") {
        auto node = std::make_unique<IRNode>(type, shape, name);
        IRNode* ptr = node.get();
        if (!name.empty()) node_map[name] = ptr;
        nodes.push_back(std::move(node));
        return ptr;
    }
    
    IRNode* get_node(const std::string& name) {
        auto it = node_map.find(name);
        return (it != node_map.end()) ? it->second : nullptr;
    }
    
    size_t node_count() const { return nodes.size(); }
    
    std::string to_string() const {
        std::ostringstream ss;
        ss << "IRGraph(" << nodes.size() << " nodes):\n";
        for (size_t i = 0; i < nodes.size(); ++i) {
            auto& n = nodes[i];
            ss << "  " << i << ": " << n->type_name();
            if (!n->name.empty()) ss << " [" << n->name << "]";
            ss << "\n";
        }
        return ss.str();
    }
    
    void clear() { nodes.clear(); node_map.clear(); }
};

class KernelLowering {
public:
    static void emit_kernel(IRNode* node) {
        if (!node) return;
        
        switch(node->type) {
            case IRType::MATMUL:
                emit_matmul(node);
                break;
            case IRType::FUSED_LINEAR_RELU:
                emit_fused_linear_relu(node);
                break;
            case IRType::FUSED_GEMM_BIAS:
                emit_fused_gemm_bias(node);
                break;
            case IRType::RELU:
                emit_relu(node);
                break;
            default:
                break;
        }
    }
    
private:
    static void emit_matmul(IRNode* node) {
        if (node->inputs.size() < 2 || node->output_tensors.empty()) return;
        
        Tensor* A = node->inputs[0]->output_tensors[0];
        Tensor* B = node->inputs[1]->output_tensors[0];
        Tensor* C = node->output_tensors[0];
        
        if (!A || !B || !C || !A->data || !B->data || !C->data) return;
        
        size_t M = A->shape[0];
        size_t K = A->shape[1];
        size_t N = B->shape[1];
        
        float* a = A->data.get() + A->offset;
        float* b = B->data.get() + B->offset;
        float* c = C->data.get() + C->offset;
        
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += a[i * K + k] * b[k * N + j];
                }
                c[i * N + j] = sum;
            }
        }
    }
    
    static void emit_fused_linear_relu(IRNode* node) {
        if (node->inputs.size() < 2) return;
        
        Tensor* A = node->inputs[0]->output_tensors[0];
        Tensor* W = node->inputs[1]->output_tensors[0];
        Tensor* C = node->output_tensors.empty() ? nullptr : node->output_tensors[0];
        
        if (!A || !W || !C) return;
        
        size_t M = A->shape[0];
        size_t K = A->shape[1];
        size_t N = W->shape[1];
        
        float* a = A->data.get() + A->offset;
        float* w = W->data.get() + W->offset;
        float* c = C->data.get() + C->offset;
        
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += a[i * K + k] * w[k * N + j];
                }
                c[i * N + j] = sum > 0.0f ? sum : 0.0f;
            }
        }
    }
    
    static void emit_fused_gemm_bias(IRNode* node) {
        if (node->inputs.size() < 3) return;
        
        Tensor* A = node->inputs[0]->output_tensors[0];
        Tensor* B = node->inputs[1]->output_tensors[0];
        Tensor* Bias = node->inputs[2]->output_tensors[0];
        Tensor* C = node->output_tensors.empty() ? nullptr : node->output_tensors[0];
        
        if (!A || !B || !Bias || !C) return;
        
        size_t M = A->shape[0];
        size_t K = A->shape[1];
        size_t N = B->shape[1];
        
        float* a = A->data.get() + A->offset;
        float* b = B->data.get() + B->offset;
        float* bias = Bias->data.get() + Bias->offset;
        float* c = C->data.get() + C->offset;
        
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = bias[j];
                for (size_t k = 0; k < K; ++k) {
                    sum += a[i * K + k] * b[k * N + j];
                }
                c[i * N + j] = sum;
            }
        }
    }
    
    static void emit_relu(IRNode* node) {
        if (node->inputs.empty() || node->output_tensors.empty()) return;
        
        Tensor* in = node->inputs[0]->output_tensors[0];
        Tensor* out = node->output_tensors[0];
        
        if (!in || !out || !in->data || !out->data) return;
        
        size_t n = in->shape.size();
        float* i_data = in->data.get() + in->offset;
        float* o_data = out->data.get() + out->offset;
        
        for (size_t j = 0; j < n; ++j) {
            o_data[j] = i_data[j] > 0.0f ? i_data[j] : 0.0f;
        }
    }
};

class IRExecutor {
public:
    static void execute(IRGraph& graph) {
        for (auto& node : graph.nodes) {
            KernelLowering::emit_kernel(node.get());
        }
    }
};

class FusionPass {
public:
    static void run(IRGraph& graph) {
        fuse_linear_relu(graph);
        fuse_matmul_bias(graph);
    }
    
private:
    static void fuse_linear_relu(IRGraph& graph) {
        for (size_t i = 0; i + 1 < graph.nodes.size(); ++i) {
            IRNode* a = graph.nodes[i].get();
            IRNode* b = graph.nodes[i + 1].get();
            
            if (a->type == IRType::MATMUL && b->type == IRType::RELU) {
                a->type = IRType::FUSED_LINEAR_RELU;
                graph.nodes.erase(graph.nodes.begin() + i + 1);
                if (i > 0) --i;
            }
        }
    }
    
    static void fuse_matmul_bias(IRGraph& graph) {
        for (size_t i = 0; i + 1 < graph.nodes.size(); ++i) {
            IRNode* a = graph.nodes[i].get();
            IRNode* b = graph.nodes[i + 1].get();
            
            if (a->type == IRType::MATMUL && b->type == IRType::ADD) {
                a->type = IRType::FUSED_GEMM_BIAS;
                graph.nodes.erase(graph.nodes.begin() + i + 1);
                if (i > 0) --i;
            }
        }
    }
};

class IRGraphOptimizer {
public:
    static void optimize(IRGraph& graph) {
        FusionPass::run(graph);
        std::cout << "[IR] Optimized to " << graph.node_count() << " nodes\n";
    }
};

inline void set_ir_debug(bool enable) {
    std::cout << "[IR] Debug " << (enable ? "enabled" : "disabled") << "\n";
}

}