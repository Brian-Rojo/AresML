#pragma once

#include <vector>
#include <string>
#include "../ir_v2/IRGraphV2.hpp"

namespace aresml {

enum class FusionType {
    NONE,
    GEMM_BIAS_RELU,
    MLP_BLOCK,
    ATTENTION_BLOCK,
    ELEMENTWISE_BLOCK,
    RESIDUAL_BLOCK,
    CUSTOM
};

struct FusionGroup {
    FusionType type;
    std::string name;
    std::vector<IRNodeV2*> nodes;
    IRNodeV2* fused_node;
    size_t fusion_score;
    
    FusionGroup() : type(FusionType::NONE), fused_node(nullptr), fusion_score(0) {}
    
    size_t original_ops() const { return nodes.size(); }
    size_t fused_ops() const { return 1; }
    size_t savings() const { return original_ops() - fused_ops(); }
    
    bool is_valid() const { return !nodes.empty() && fused_node != nullptr; }
};

struct FusionResult {
    std::vector<FusionGroup> groups;
    size_t original_nodes;
    size_t fused_nodes;
    size_t total_savings;
    float estimated_speedup;
    
    FusionResult() : original_nodes(0), fused_nodes(0), total_savings(0), estimated_speedup(1.0f) {}
    
    std::string to_string() const {
        std::string s = "FusionResult:\n";
        s += "  Original nodes: " + std::to_string(original_nodes) + "\n";
        s += "  Fused nodes: " + std::to_string(fused_nodes) + "\n";
        s += "  Savings: " + std::to_string(total_savings) + "\n";
        s += "  Estimated speedup: " + std::to_string(estimated_speedup) + "x\n";
        s += "  Groups: " + std::to_string(groups.size()) + "\n";
        return s;
    }
};

struct FusionConstraint {
    size_t max_fusion_size;
    size_t min_fusion_size;
    bool allow_memory_pressure;
    float memory_budget_mb;
    
    FusionConstraint() 
        : max_fusion_size(8)
        , min_fusion_size(2)
        , allow_memory_pressure(true)
        , memory_budget_mb(1024.0f) {}
};

class FusionGroups {
public:
    static FusionGroup create_gemm_bias_relu_group(const std::vector<IRNodeV2*>& nodes) {
        FusionGroup group;
        if (nodes.size() >= 3) {
            group.type = FusionType::GEMM_BIAS_RELU;
            group.name = "GEMM_BIAS_RELU";
            group.nodes = nodes;
            group.fused_node = nodes.front();
            group.fusion_score = 100;
        }
        return group;
    }
    
    static FusionGroup create_mlp_block_group(const std::vector<IRNodeV2*>& nodes) {
        FusionGroup group;
        if (nodes.size() >= 4) {
            group.type = FusionType::MLP_BLOCK;
            group.name = "MLP_BLOCK";
            group.nodes = nodes;
            group.fused_node = nodes.front();
            group.fusion_score = 150;
        }
        return group;
    }
    
    static FusionGroup create_attention_block_group(const std::vector<IRNodeV2*>& nodes) {
        FusionGroup group;
        if (nodes.size() >= 3) {
            group.type = FusionType::ATTENTION_BLOCK;
            group.name = "ATTENTION_BLOCK";
            group.nodes = nodes;
            group.fused_node = nodes.front();
            group.fusion_score = 200;
        }
        return group;
    }
    
    static FusionGroup create_elementwise_block_group(const std::vector<IRNodeV2*>& nodes) {
        FusionGroup group;
        if (nodes.size() >= 2) {
            group.type = FusionType::ELEMENTWISE_BLOCK;
            group.name = "ELEMENTWISE_BLOCK";
            group.nodes = nodes;
            group.fused_node = nodes.front();
            group.fusion_score = 50;
        }
        return group;
    }
    
    static FusionGroup create_residual_block_group(const std::vector<IRNodeV2*>& nodes) {
        FusionGroup group;
        if (nodes.size() >= 2) {
            group.type = FusionType::RESIDUAL_BLOCK;
            group.name = "RESIDUAL_BLOCK";
            group.nodes = nodes;
            group.fused_node = nodes.front();
            group.fusion_score = 80;
        }
        return group;
    }
    
    static bool can_fuse(const FusionGroup& group, const FusionConstraint& constraint) {
        if (!group.is_valid()) return false;
        if (group.nodes.size() < constraint.min_fusion_size) return false;
        if (group.nodes.size() > constraint.max_fusion_size) return false;
        return true;
    }
    
    static float estimate_speedup(const FusionGroup& group) {
        switch(group.type) {
            case FusionType::GEMM_BIAS_RELU:
                return 2.5f;
            case FusionType::MLP_BLOCK:
                return 3.0f;
            case FusionType::ATTENTION_BLOCK:
                return 4.0f;
            case FusionType::ELEMENTWISE_BLOCK:
                return 1.8f;
            case FusionType::RESIDUAL_BLOCK:
                return 1.5f;
            default:
                return 1.0f;
        }
    }
};

}
