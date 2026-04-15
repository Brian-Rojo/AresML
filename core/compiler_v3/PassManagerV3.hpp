#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <functional>
#include <unordered_set>
#include "../ir_v3/IRGraphV3.hpp"
#include "GraphLoweringV3.hpp"

namespace aresml {

class PassManagerV3 {
public:
    using PassFn = std::function<void(IRGraphV3&)>;
    
    PassManagerV3() {
        register_standard_passes();
    }
    
    void register_pass(const std::string& name, PassFn pass_fn) {
        passes_[name] = pass_fn;
    }
    
    void run_passes(IRGraphV3& graph) {
        for (auto& name : pass_order_) {
            auto it = passes_.find(name);
            if (it != passes_.end()) {
                it->second(graph);
            }
        }
    }
    
    void run_until_fixed_point(IRGraphV3& graph, size_t max_iterations = 10) {
        size_t prev_count = graph.instruction_count();
        
        for (size_t iter = 0; iter < max_iterations; ++iter) {
            run_passes(graph);
            
            if (graph.instruction_count() == prev_count) {
                break;
            }
            prev_count = graph.instruction_count();
        }
    }
    
private:
    std::unordered_map<std::string, PassFn> passes_;
    std::vector<std::string> pass_order_;
    
    void register_standard_passes() {
        register_pass("inline_small_ops", inline_small_ops);
        register_pass("remove_dead_code", remove_dead_code);
        register_pass("fuse_elementwise", fuse_elementwise);
        register_pass("allocate_registers", allocate_registers);
        
        pass_order_ = {
            "inline_small_ops",
            "fuse_elementwise",
            "remove_dead_code",
            "allocate_registers"
        };
    }
    
    static void inline_small_ops(IRGraphV3& graph) {
        std::vector<std::unique_ptr<IRNodeV3>> inlined;
        
        for (auto& inst : graph.instructions) {
            if (inst->estimated_cycles < 100 && !inst->is_fused) {
                inst->is_inlined = true;
            }
            inlined.push_back(std::move(inst));
        }
        
        graph.instructions = std::move(inlined);
    }
    
    static void remove_dead_code(IRGraphV3& graph) {
        std::unordered_set<std::string> live_values;
        
        for (const auto& out : graph.outputs) {
            live_values.insert(out);
        }
        
        std::vector<std::unique_ptr<IRNodeV3>> filtered;
        for (auto& inst : graph.instructions) {
            bool has_live_output = false;
            for (const auto& out : inst->outputs) {
                if (live_values.count(out.prefix)) {
                    has_live_output = true;
                    break;
                }
            }
            
            if (has_live_output) {
                for (const auto& in : inst->inputs) {
                    live_values.insert(in.prefix);
                }
                filtered.push_back(std::move(inst));
            }
        }
        
        graph.instructions = std::move(filtered);
    }
    
    static void fuse_elementwise(IRGraphV3& graph) {
        for (size_t i = 0; i + 1 < graph.instructions.size(); ++i) {
            auto* a = graph.instructions[i].get();
            auto* b = graph.instructions[i + 1].get();
            
            if ((a->op == IROpV3::ELEMENTWISE_ADD || a->op == IROpV3::ELEMENTWISE_MUL) &&
                (b->op == IROpV3::ELEMENTWISE_ADD || b->op == IROpV3::ELEMENTWISE_MUL)) {
                a->is_fused = true;
                b->is_inlined = true;
            }
        }
    }
    
    static void allocate_registers(IRGraphV3& graph) {
        uint32_t reg = 0;
        for (auto& inst : graph.instructions) {
            for (auto& out : inst->outputs) {
                out.version = reg++;
            }
        }
    }
};

inline void run_standard_passes(IRGraphV3& graph) {
    PassManagerV3 pm;
    pm.run_until_fixed_point(graph);
}

}
