#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include "../ir_v2/IRGraphV2.hpp"

namespace aresml {

struct FusionPattern {
    std::string name;
    std::vector<IROpV2> sequence;
    IROpV2 fused_op;
    
    bool matches(const std::vector<IRNodeV2*>& nodes) const {
        if (nodes.size() != sequence.size()) return false;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i]->op != sequence[i]) return false;
        }
        return true;
    }
};

class PatternMatcher {
public:
    static std::vector<FusionPattern> get_patterns() {
        return {
            {"GEMM_BIAS_RELU", {IROpV2::MATMUL, IROpV2::BIAS_ADD, IROpV2::RELU}, IROpV2::FUSED_GEMM_BIAS_RELU},
            {"GEMM_BIAS", {IROpV2::MATMUL, IROpV2::BIAS_ADD}, IROpV2::FUSED_GEMM_BIAS},
            {"MLP_BLOCK", {IROpV2::MATMUL, IROpV2::RELU, IROpV2::MATMUL, IROpV2::RELU}, IROpV2::FUSED_MLP_BLOCK},
            {"ATTENTION_BLOCK", {IROpV2::ATTENTION_QKV, IROpV2::MATMUL, IROpV2::SOFTMAX}, IROpV2::FUSED_ATTENTION_BLOCK},
            {"ELEMWISE_BLOCK", {IROpV2::ADD, IROpV2::MUL}, IROpV2::FUSED_ELEMENTWISE_BLOCK},
            {"RESIDUAL_BLOCK", {IROpV2::ADD, IROpV2::RELU}, IROpV2::FUSED_RESIDUAL_BLOCK},
        };
    }
    
    static std::vector<std::vector<IRNodeV2*>> find_patterns(IRGraphV2& graph) {
        std::vector<std::vector<IRNodeV2*>> matches;
        auto patterns = get_patterns();
        
        for (size_t i = 0; i + 1 < graph.nodes.size(); ++i) {
            for (auto& pattern : patterns) {
                if (pattern.sequence.size() >= 2 && i + pattern.sequence.size() <= graph.nodes.size()) {
                    std::vector<IRNodeV2*> candidate;
                    for (size_t j = 0; j < pattern.sequence.size(); ++j) {
                        candidate.push_back(graph.nodes[i + j].get());
                    }
                    if (pattern.matches(candidate)) {
                        matches.push_back(candidate);
                    }
                }
            }
        }
        
        return matches;
    }
    
    static std::vector<std::vector<IRNodeV2*>> find_gemm_bias_relu(IRGraphV2& graph) {
        std::vector<std::vector<IRNodeV2*>> result;
        
        for (size_t i = 0; i + 2 < graph.nodes.size(); ++i) {
            auto* a = graph.nodes[i].get();
            auto* b = graph.nodes[i + 1].get();
            auto* c = graph.nodes[i + 2].get();
            
            if (a->op == IROpV2::MATMUL && 
                b->op == IROpV2::BIAS_ADD && 
                c->op == IROpV2::RELU) {
                result.push_back({a, b, c});
            }
        }
        
        return result;
    }
    
    static std::vector<std::vector<IRNodeV2*>> find_mlp_blocks(IRGraphV2& graph) {
        std::vector<std::vector<IRNodeV2*>> result;
        
        for (size_t i = 0; i + 3 < graph.nodes.size(); ++i) {
            auto* a = graph.nodes[i].get();
            auto* b = graph.nodes[i + 1].get();
            auto* c = graph.nodes[i + 2].get();
            auto* d = graph.nodes[i + 3].get();
            
            if (a->op == IROpV2::MATMUL && 
                b->op == IROpV2::RELU && 
                c->op == IROpV2::MATMUL && 
                d->op == IROpV2::RELU) {
                result.push_back({a, b, c, d});
            }
        }
        
        return result;
    }
    
    static std::vector<std::vector<IRNodeV2*>> find_attention_blocks(IRGraphV2& graph) {
        std::vector<std::vector<IRNodeV2*>> result;
        
        for (size_t i = 0; i + 2 < graph.nodes.size(); ++i) {
            auto* a = graph.nodes[i].get();
            auto* b = graph.nodes[i + 1].get();
            auto* c = graph.nodes[i + 2].get();
            
            if (a->op == IROpV2::ATTENTION_QKV && 
                b->op == IROpV2::MATMUL && 
                c->op == IROpV2::SOFTMAX) {
                result.push_back({a, b, c});
            }
        }
        
        return result;
    }
    
    static std::vector<std::vector<IRNodeV2*>> find_elementwise_chains(IRGraphV2& graph) {
        std::vector<std::vector<IRNodeV2*>> result;
        
        for (size_t i = 0; i + 1 < graph.nodes.size(); ++i) {
            auto* a = graph.nodes[i].get();
            auto* b = graph.nodes[i + 1].get();
            
            if ((a->op == IROpV2::ADD || a->op == IROpV2::MUL || a->op == IROpV2::SUB) &&
                (b->op == IROpV2::ADD || b->op == IROpV2::MUL || b->op == IROpV2::SUB)) {
                result.push_back({a, b});
            }
        }
        
        return result;
    }
};

}
