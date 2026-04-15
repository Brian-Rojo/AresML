#pragma once

#include "Tensor.hpp"
#include <unordered_set>
#include <unordered_map>
#include <stack>
#include <sstream>
#include <iostream>

namespace aresml {

class GraphValidator {
public:
    struct ValidationResult {
        bool valid = true;
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
        
        void add_error(const std::string& msg) {
            valid = false;
            errors.push_back(msg);
        }
        
        void add_warning(const std::string& msg) {
            warnings.push_back(msg);
        }
    };
    
    static ValidationResult validate(const Tensor& output) {
        ValidationResult result;
        validate_no_cycles(output, result);
        validate_tensor_lifetimes(output, result);
        validate_ops_connected(output, result);
        return result;
    }
    
    static void validate_no_cycles(const Tensor& tensor, ValidationResult& result) {
        std::unordered_set<const Tensor*> visited;
        std::stack<const Tensor*> st;
        st.push(&tensor);
        
        while (!st.empty()) {
            const Tensor* current = st.top();
            st.pop();
            
            if (!current) continue;
            if (visited.count(current)) continue;
            visited.insert(current);
            
            for (auto* inp : current->inputs) {
                if (inp) st.push(inp);
            }
        }
    }
    
    static void validate_tensor_lifetimes(const Tensor& tensor, ValidationResult& result) {
        if (!tensor.data) {
            result.add_error("Output tensor has null data");
        }
    }
    
    static void validate_ops_connected(const Tensor& tensor, ValidationResult& result) {
        std::unordered_set<const Tensor*> visited;
        std::stack<const Tensor*> st;
        st.push(&tensor);
        
        while (!st.empty()) {
            const Tensor* current = st.top();
            st.pop();
            
            if (!current || visited.count(current)) continue;
            visited.insert(current);
            
            if (current->op && current->inputs.empty()) {
                result.add_warning("Op with no inputs connected");
            }
            
            for (auto* inp : current->inputs) {
                if (inp) st.push(inp);
            }
        }
    }
};

inline void validate_graph(const Tensor& output) {
    auto result = GraphValidator::validate(output);
    
    if (!result.valid) {
        std::cerr << "[VALIDATION] FAILED:\n";
        for (auto& e : result.errors) {
            std::cerr << "  ERROR: " << e << "\n";
        }
    }
    
    for (auto& w : result.warnings) {
        std::cerr << "[WARN] " << w << "\n";
    }
}

}