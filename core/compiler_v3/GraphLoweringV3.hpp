#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <functional>
#include "../ir_v2/IRGraphV2.hpp"
#include "../ir_v3/IRGraphV3.hpp"
#include "../ir_v3/IRNodeV3.hpp"

namespace aresml {

struct LoweringResult {
    std::unique_ptr<IRGraphV3> graph;
    size_t instructions_eliminated;
    size_t ops_fused;
    std::string hash;
    
    LoweringResult() : instructions_eliminated(0), ops_fused(0) {}
};

class GraphLoweringV3 {
public:
    GraphLoweringV3() {}
    
    LoweringResult lower(IRGraphV2& v2_graph) {
        LoweringResult result;
        result.graph = std::make_unique<IRGraphV3>();
        
        for (const auto& input : v2_graph.get_inputs()) {
            result.graph->add_input(input->name);
        }
        
        for (const auto& param : v2_graph.get_parameters()) {
            result.graph->add_parameter(param->name);
        }
        
        for (auto& node : v2_graph.nodes) {
            auto* v3_node = convert_node(node.get());
            if (v3_node) {
                result.graph->instructions.push_back(std::unique_ptr<IRNodeV3>(v3_node));
            }
        }
        
        for (auto& node : v2_graph.nodes) {
            if (node->op == IROpV2::MATMUL) {
                auto fused = try_fuse_linear_chain(result.graph.get(), node.get(), v2_graph);
                if (fused) result.ops_fused++;
            }
        }
        
        result.instructions_eliminated = v2_graph.node_count() - result.graph->instruction_count();
        result.hash = result.graph->compute_hash();
        
        return result;
    }
    
private:
    IRNodeV3* convert_node(IRNodeV2* v2_node) {
        if (!v2_node) return nullptr;
        
        IROpV3 v3_op = IROpV3::UNKNOWN;
        
        switch(v2_node->op) {
            case IROpV2::INPUT:
            case IROpV2::PARAM:
            case IROpV2::WEIGHT:
                return nullptr;
            case IROpV2::MATMUL:
                v3_op = IROpV3::GEMM;
                break;
            case IROpV2::BIAS_ADD:
                v3_op = IROpV3::ELEMENTWISE_ADD;
                break;
            case IROpV2::ADD:
                v3_op = IROpV3::ELEMENTWISE_ADD;
                break;
            case IROpV2::MUL:
                v3_op = IROpV3::ELEMENTWISE_MUL;
                break;
            case IROpV2::RELU:
                v3_op = IROpV3::ELEMENTWISE_RELU;
                break;
            case IROpV2::GELU:
                v3_op = IROpV3::ELEMENTWISE_GELU;
                break;
            case IROpV2::SOFTMAX:
                v3_op = IROpV3::SOFTMAX;
                break;
            case IROpV2::FUSED_GEMM_BIAS_RELU:
                v3_op = IROpV3::GEMM_BIAS_RELU;
                break;
            case IROpV2::FUSED_MLP_BLOCK:
                v3_op = IROpV3::FUSED_LINEAR_CHAIN;
                break;
            default:
                return nullptr;
        }
        
        auto* v3_node = new IRNodeV3(v2_node->id, v2_node->name, v3_op);
        v3_node->shape = v2_node->shape.dims;
        
        for (size_t i = 0; i < v2_node->operands.size(); ++i) {
            v3_node->add_input(SSAName("v" + std::to_string(i), 0));
        }
        
        v3_node->add_output(SSAName(v2_node->name, 0));
        
        return v3_node;
    }
    
    bool try_fuse_linear_chain(IRGraphV3* v3_graph, IRNodeV2* matmul_node, IRGraphV2& v2_graph) {
        return false;
    }
};

}
