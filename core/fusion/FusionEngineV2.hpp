#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <algorithm>
#include "PatternMatcher.hpp"
#include "FusionGroups.hpp"
#include "../ir_v2/IRGraphV2.hpp"

namespace aresml {

class FusionEngineV2 {
public:
    FusionEngineV2() {
        constraint_ = FusionConstraint();
    }
    
    explicit FusionEngineV2(const FusionConstraint& constraint) 
        : constraint_(constraint) {}
    
    FusionResult run(IRGraphV2& graph) {
        FusionResult result;
        result.original_nodes = graph.node_count();
        
        if (graph.nodes.empty()) {
            result.fused_nodes = 0;
            return result;
        }
        
        auto gemm_groups = PatternMatcher::find_gemm_bias_relu(graph);
        for (auto& nodes : gemm_groups) {
            auto group = FusionGroups::create_gemm_bias_relu_group(nodes);
            if (FusionGroups::can_fuse(group, constraint_)) {
                apply_fusion(graph, group);
                result.groups.push_back(group);
            }
        }
        
        auto mlp_groups = PatternMatcher::find_mlp_blocks(graph);
        for (auto& nodes : mlp_groups) {
            auto group = FusionGroups::create_mlp_block_group(nodes);
            if (FusionGroups::can_fuse(group, constraint_)) {
                apply_fusion(graph, group);
                result.groups.push_back(group);
            }
        }
        
        auto attention_groups = PatternMatcher::find_attention_blocks(graph);
        for (auto& nodes : attention_groups) {
            auto group = FusionGroups::create_attention_block_group(nodes);
            if (FusionGroups::can_fuse(group, constraint_)) {
                apply_fusion(graph, group);
                result.groups.push_back(group);
            }
        }
        
        auto elem_groups = PatternMatcher::find_elementwise_chains(graph);
        for (auto& nodes : elem_groups) {
            auto group = FusionGroups::create_elementwise_block_group(nodes);
            if (FusionGroups::can_fuse(group, constraint_)) {
                apply_fusion(graph, group);
                result.groups.push_back(group);
            }
        }
        
        result.fused_nodes = graph.node_count();
        result.total_savings = result.original_nodes - result.fused_nodes;
        
        float total_speedup = 1.0f;
        for (auto& g : result.groups) {
            total_speedup += FusionGroups::estimate_speedup(g) - 1.0f;
        }
        result.estimated_speedup = total_speedup;
        
        return result;
    }
    
    void set_constraint(const FusionConstraint& constraint) {
        constraint_ = constraint;
    }
    
    const FusionConstraint& get_constraint() const {
        return constraint_;
    }
    
private:
    FusionConstraint constraint_;
    
    void apply_fusion(IRGraphV2& graph, FusionGroup& group) {
        if (group.nodes.empty()) return;
        
        IRNodeV2* first = group.nodes.front();
        IRNodeV2* last = group.nodes.back();
        
        IROpV2 fused_op = IROpV2::UNKNOWN;
        
        switch(group.type) {
            case FusionType::GEMM_BIAS_RELU:
                fused_op = IROpV2::FUSED_GEMM_BIAS_RELU;
                break;
            case FusionType::MLP_BLOCK:
                fused_op = IROpV2::FUSED_MLP_BLOCK;
                break;
            case FusionType::ATTENTION_BLOCK:
                fused_op = IROpV2::FUSED_ATTENTION_BLOCK;
                break;
            case FusionType::ELEMENTWISE_BLOCK:
                fused_op = IROpV2::FUSED_ELEMENTWISE_BLOCK;
                break;
            case FusionType::RESIDUAL_BLOCK:
                fused_op = IROpV2::FUSED_RESIDUAL_BLOCK;
                break;
            default:
                return;
        }
        
        first->op = fused_op;
        
        for (size_t i = 1; i < group.nodes.size(); ++i) {
            group.nodes[i]->op = IROpV2::UNKNOWN;
        }
        
        for (auto it = graph.nodes.begin(); it != graph.nodes.end(); ) {
            if ((*it)->op == IROpV2::UNKNOWN) {
                it = graph.nodes.erase(it);
            } else {
                ++it;
            }
        }
    }
};

inline FusionResult fuse_ir_graph(IRGraphV2& graph) {
    FusionEngineV2 engine;
    return engine.run(graph);
}

inline FusionResult fuse_ir_graph(IRGraphV2& graph, const FusionConstraint& constraint) {
    FusionEngineV2 engine(constraint);
    return engine.run(graph);
}

}
