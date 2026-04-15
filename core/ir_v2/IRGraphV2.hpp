#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <string>
#include <sstream>
#include <functional>
#include "IRNodeV2.hpp"

namespace aresml {

class IRGraphV2 {
public:
    std::vector<std::unique_ptr<IRNodeV2>> nodes;
    std::vector<std::unique_ptr<IRValue>> values;
    std::unordered_map<std::string, IRNodeV2*> node_map;
    std::unordered_map<std::string, IRValue*> value_map;
    
    size_t next_node_id = 0;
    size_t next_value_id = 0;
    
    IRNodeV2* add_node(IROpV2 op, const IRShape& shape, const std::string& name = "") {
        if (name.empty()) {
            return add_node(op, shape, "_v" + std::to_string(next_node_id++));
        }
        
        auto node = std::make_unique<IRNodeV2>(next_node_id++, name, op, shape);
        IRNodeV2* ptr = node.get();
        node_map[name] = ptr;
        nodes.push_back(std::move(node));
        return ptr;
    }
    
    IRValue* add_value(const std::string& name, IROpV2 op, const IRShape& shape) {
        auto value = std::make_unique<IRValue>(name, op, shape);
        IRValue* ptr = value.get();
        value_map[name] = ptr;
        values.push_back(std::move(value));
        return ptr;
    }
    
    IRNodeV2* get_node(const std::string& name) {
        auto it = node_map.find(name);
        return (it != node_map.end()) ? it->second : nullptr;
    }
    
    IRValue* get_value(const std::string& name) {
        auto it = value_map.find(name);
        return (it != value_map.end()) ? it->second : nullptr;
    }
    
    void add_edge(IRNodeV2* from, IRValue* result, IRNodeV2* to, size_t operand_index = 0) {
        if (from && result && to) {
            from->add_result(result);
            to->add_operand(result, operand_index);
        }
    }
    
    size_t node_count() const { return nodes.size(); }
    size_t value_count() const { return values.size(); }
    
    std::vector<IRNodeV2*> get_parameters() {
        std::vector<IRNodeV2*> params;
        for (auto& n : nodes) {
            if (n->is_param()) {
                params.push_back(n.get());
            }
        }
        return params;
    }
    
    std::vector<IRNodeV2*> get_inputs() {
        std::vector<IRNodeV2*> inputs;
        for (auto& n : nodes) {
            if (n->op == IROpV2::INPUT) {
                inputs.push_back(n.get());
            }
        }
        return inputs;
    }
    
    std::vector<IRNodeV2*> get_leaves() {
        std::vector<IRNodeV2*> leaves;
        for (auto& n : nodes) {
            if (n->is_leaf()) {
                leaves.push_back(n.get());
            }
        }
        return leaves;
    }
    
    std::string to_string() const {
        std::ostringstream ss;
        ss << "IRGraphV2(" << nodes.size() << " nodes, " << values.size() << " values):\n";
        for (auto& n : nodes) {
            ss << "  " << n->to_string() << "\n";
        }
        return ss.str();
    }
    
    void clear() {
        nodes.clear();
        values.clear();
        node_map.clear();
        value_map.clear();
        next_node_id = 0;
        next_value_id = 0;
    }
    
    void topological_sort(std::vector<IRNodeV2*>& out) {
        std::unordered_set<IRNodeV2*> visited;
        
        std::function<void(IRNodeV2*)> dfs = [&](IRNodeV2* node) {
            if (visited.count(node)) return;
            visited.insert(node);
            
            for (auto& use : node->operands) {
                if (use.value && use.value != nullptr) {
                    for (auto& n : nodes) {
                        for (auto* r : n->results) {
                            if (r == use.value) {
                                dfs(n.get());
                                break;
                            }
                        }
                    }
                }
            }
            
            out.push_back(node);
        };
        
        for (auto& n : nodes) {
            dfs(n.get());
        }
        
        std::reverse(out.begin(), out.end());
    }
};

inline IRGraphV2* create_ir_graph() {
    return new IRGraphV2();
}

inline void destroy_ir_graph(IRGraphV2* graph) {
    delete graph;
}

}
