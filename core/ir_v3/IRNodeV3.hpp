#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "../ir_v2/IRGraphV2.hpp"

namespace aresml {

enum class IROpV3 {
    UNKNOWN,
    PARAM,
    INPUT,
    CONSTANT,
    GEMM,
    GEMM_BIAS_RELU,
    ELEMENTWISE_ADD,
    ELEMENTWISE_MUL,
    ELEMENTWISE_RELU,
    ELEMENTWISE_GELU,
    SOFTMAX,
    REDUCE_SUM,
    BROADCAST,
    SLICE,
    CONCAT,
    COPY,
    FUSED_LINEAR_CHAIN,
    CUSTOM
};

enum class IRLayoutV3 {
    ROW_MAJOR,
    COL_MAJOR,
    BLOCKED,
    TEMPORAL
};

struct SSAName {
    std::string prefix;
    uint32_t version;
    
    SSAName() : version(0) {}
    SSAName(const std::string& p, uint32_t v) : prefix(p), version(v) {}
    
    std::string to_string() const {
        return prefix + "." + std::to_string(version);
    }
    
    SSAName next() const {
        return SSAName(prefix, version + 1);
    }
};

struct IRValueV3 {
    SSAName name;
    IROpV3 op;
    std::vector<int64_t> shape;
    IRLayoutV3 layout;
    size_t offset;
    bool is_constant;
    float constant_value;
    
    IRValueV3() : op(IROpV3::UNKNOWN), layout(IRLayoutV3::ROW_MAJOR), offset(0),
                  is_constant(false), constant_value(0.0f) {}
    
    IRValueV3(const SSAName& n, IROpV3 o) : name(n), op(o), layout(IRLayoutV3::ROW_MAJOR),
                                            offset(0), is_constant(false), constant_value(0.0f) {}
    
    size_t numel() const {
        size_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
    
    size_t bytes() const {
        return numel() * sizeof(float);
    }
};

struct IRInstructionV3 {
    size_t id;
    IROpV3 op;
    std::vector<SSAName> operands;
    std::vector<SSAName> results;
    std::unordered_map<std::string, std::string> attrs;
    size_t estimated_cycles;
    
    IRInstructionV3() : id(0), op(IROpV3::UNKNOWN), estimated_cycles(0) {}
    IRInstructionV3(size_t i, IROpV3 o) : id(i), op(o), estimated_cycles(0) {}
    
    bool is_terminal() const {
        return op == IROpV3::SOFTMAX || op == IROpV3::REDUCE_SUM;
    }
    
    bool is_pure() const {
        return op != IROpV3::SOFTMAX && op != IROpV3::REDUCE_SUM;
    }
    
    std::string op_name() const {
        switch(op) {
            case IROpV3::GEMM: return "gemm";
            case IROpV3::GEMM_BIAS_RELU: return "gemm_bias_relu";
            case IROpV3::ELEMENTWISE_ADD: return "add";
            case IROpV3::ELEMENTWISE_MUL: return "mul";
            case IROpV3::ELEMENTWISE_RELU: return "relu";
            case IROpV3::ELEMENTWISE_GELU: return "gelu";
            case IROpV3::SOFTMAX: return "softmax";
            case IROpV3::REDUCE_SUM: return "sum";
            default: return "unknown";
        }
    }
};

class IRNodeV3 {
public:
    size_t id;
    std::string original_name;
    IROpV3 op;
    std::vector<SSAName> inputs;
    std::vector<SSAName> outputs;
    std::vector<int64_t> shape;
    IRLayoutV3 layout;
    size_t estimated_cycles;
    bool is_fused;
    bool is_inlined;
    
    IRNodeV3() : id(0), op(IROpV3::UNKNOWN), estimated_cycles(0),
                 is_fused(false), is_inlined(false) {}
    
    IRNodeV3(size_t i, const std::string& name, IROpV3 o)
        : id(i), original_name(name), op(o), estimated_cycles(0),
          is_fused(false), is_inlined(false) {}
    
    void add_input(const SSAName& name) { inputs.push_back(name); }
    void add_output(const SSAName& name) { outputs.push_back(name); }
    
    bool has_cycles_estimate() const { return estimated_cycles > 0; }
    
    std::string to_string() const {
        return original_name + " = " + op_name() + "()";
    }
    
private:
    std::string op_name() const {
        switch(op) {
            case IROpV3::GEMM: return "gemm";
            case IROpV3::GEMM_BIAS_RELU: return "gemm_bias_relu";
            case IROpV3::ELEMENTWISE_ADD: return "add";
            case IROpV3::ELEMENTWISE_MUL: return "mul";
            case IROpV3::ELEMENTWISE_RELU: return "relu";
            case IROpV3::ELEMENTWISE_GELU: return "gelu";
            case IROpV3::SOFTMAX: return "softmax";
            case IROpV3::REDUCE_SUM: return "sum";
            case IROpV3::FUSED_LINEAR_CHAIN: return "fused_linear_chain";
            default: return "unknown";
        }
    }
};

}
