#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <string>
#include <sstream>
#include <functional>
#include "IRNodeV3.hpp"

namespace aresml {

class IRGraphV3 {
public:
    std::vector<std::unique_ptr<IRNodeV3>> instructions;
    std::unordered_map<std::string, SSAName> value_map;
    std::unordered_map<std::string, IRNodeV3*> node_map;
    
    size_t next_id = 0;
    uint32_t ssa_version = 0;
    
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::string> parameters;
    
    IRNodeV3* add_instruction(IROpV3 op, const std::string& name = "") {
        if (name.empty()) {
            return add_instruction(op, "_v" + std::to_string(next_id++));
        }
        
        auto node = std::make_unique<IRNodeV3>(next_id++, name, op);
        IRNodeV3* ptr = node.get();
        node_map[name] = ptr;
        instructions.push_back(std::move(node));
        return ptr;
    }
    
    IRNodeV3* get_instruction(const std::string& name) {
        auto it = node_map.find(name);
        return (it != node_map.end()) ? it->second : nullptr;
    }
    
    SSAName create_ssa(const std::string& prefix) {
        return SSAName(prefix, ssa_version++);
    }
    
    void add_input(const std::string& name) {
        inputs.push_back(name);
    }
    
    void add_output(const std::string& name) {
        outputs.push_back(name);
    }
    
    void add_parameter(const std::string& name) {
        parameters.push_back(name);
    }
    
    size_t instruction_count() const { return instructions.size(); }
    
    std::string to_string() const {
        std::ostringstream ss;
        ss << "IRGraphV3(" << instructions.size() << " instructions):\n";
        ss << "  Inputs: " << inputs.size() << "\n";
        ss << "  Outputs: " << outputs.size() << "\n";
        ss << "  Parameters: " << parameters.size() << "\n";
        ss << "Instructions:\n";
        for (auto& inst : instructions) {
            ss << "  " << inst->to_string() << "\n";
        }
        return ss.str();
    }
    
    void clear() {
        instructions.clear();
        node_map.clear();
        value_map.clear();
        inputs.clear();
        outputs.clear();
        parameters.clear();
        next_id = 0;
        ssa_version = 0;
    }
    
    std::string compute_hash() const {
        std::ostringstream ss;
        ss << instruction_count() << "|";
        for (auto& inst : instructions) {
            ss << static_cast<int>(inst->op) << ";";
            for (auto& in : inst->inputs) {
                ss << in.to_string() << ",";
            }
            ss << "|";
            for (auto& shape : inst->shape) {
                ss << shape << ",";
            }
        }
        return ss.str();
    }
    
    void topological_sort() {
        std::unordered_set<IRNodeV3*> visited;
        std::vector<std::unique_ptr<IRNodeV3>> sorted;
        
        std::function<void(IRNodeV3*)> dfs = [&](IRNodeV3* node) {
            if (!node || visited.count(node)) return;
            visited.insert(node);
            
            for (auto& input_name : node->inputs) {
                auto it = node_map.find(input_name.prefix);
                if (it != node_map.end()) {
                    dfs(it->second);
                }
            }
            
            sorted.push_back(std::unique_ptr<IRNodeV3>(node));
        };
        
        for (auto& inst : instructions) {
            dfs(inst.get());
        }
        
        std::reverse(sorted.begin(), sorted.end());
        instructions = std::move(sorted);
    }
};

}
