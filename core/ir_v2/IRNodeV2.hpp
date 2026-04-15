#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "../../core/Tensor.hpp"

namespace aresml {

enum class IROpV2 {
    UNKNOWN,
    INPUT,
    PARAM,
    WEIGHT,
    MATMUL,
    ADD,
    MUL,
    SUB,
    DIV,
    BIAS_ADD,
    RELU,
    SILU,
    GELU,
    SOFTMAX,
    LOG_SOFTMAX,
    LAYERNORM,
    RMSNORM,
    ATTENTION_QKV,
    ATTENTION_SCORE,
    ATTENTION_WEIGHT,
    TRANSPOSE,
    VIEW,
    RESHAPE,
    CONTIGUOUS,
    SUM,
    MEAN,
    MAX,
    CLIP,
    DROPOUT_FORWARD,
    DROPOUT_BACKWARD,
    CONCAT,
    SPLIT,
    SLICE,
    FUSED_GEMM_BIAS_RELU,
    FUSED_GEMM_BIAS,
    FUSED_ATTENTION_BLOCK,
    FUSED_MLP_BLOCK,
    FUSED_ELEMENTWISE_BLOCK,
    FUSED_RESIDUAL_BLOCK,
    OUTPUT
};

enum class IRDType {
    FLOAT32,
    FLOAT16,
    BFLOAT16,
    INT32,
    INT8
};

enum class IRLayout {
    ROW_MAJOR,
    COL_MAJOR,
    BLOCKED
};

struct IRShape {
    std::vector<int64_t> dims;
    
    IRShape() = default;
    IRShape(const std::vector<int64_t>& d) : dims(d) {}
    IRShape(int64_t d0) : dims{d0} {}
    IRShape(int64_t d0, int64_t d1) : dims{d0, d1} {}
    IRShape(int64_t d0, int64_t d1, int64_t d2) : dims{d0, d1, d2} {}
    IRShape(int64_t d0, int64_t d1, int64_t d2, int64_t d3) : dims{d0, d1, d2, d3} {}
    
    size_t size() const { return dims.size(); }
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
    bool operator==(const IRShape& other) const { return dims == other.dims; }
};

struct IRValue {
    std::string name;
    IROpV2 op;
    IRShape shape;
    IRDType dtype;
    bool is_constant = false;
    std::vector<float> constant_data;
    
    IRValue() = default;
    IRValue(const std::string& n, IROpV2 o, const IRShape& s)
        : name(n), op(o), shape(s) {}
};

struct IRUse {
    IRValue* value;
    size_t index;
    
    IRUse() : value(nullptr), index(0) {}
    IRUse(IRValue* v, size_t i) : value(v), index(i) {}
};

class IRNodeV2 {
public:
    size_t id;
    std::string name;
    IROpV2 op;
    IRShape shape;
    IRDType dtype = IRDType::FLOAT32;
    IRLayout layout = IRLayout::ROW_MAJOR;
    
    std::vector<IRUse> operands;
    std::vector<IRValue*> results;
    
    std::unordered_map<std::string, std::string> attributes;
    
    bool is_fused() const;
    bool is_leaf() const;
    bool is_param() const;
    
    std::string op_name() const;
    std::string to_string() const;
    
    IRNodeV2() = default;
    IRNodeV2(size_t i, const std::string& n, IROpV2 o);
    IRNodeV2(size_t i, const std::string& n, IROpV2 o, const IRShape& s);
    
    void add_operand(IRValue* val, size_t index = 0);
    void add_result(IRValue* val);
};

inline bool IRNodeV2::is_fused() const {
    return op == IROpV2::FUSED_GEMM_BIAS_RELU ||
           op == IROpV2::FUSED_GEMM_BIAS ||
           op == IROpV2::FUSED_ATTENTION_BLOCK ||
           op == IROpV2::FUSED_MLP_BLOCK ||
           op == IROpV2::FUSED_ELEMENTWISE_BLOCK ||
           op == IROpV2::FUSED_RESIDUAL_BLOCK;
}

inline bool IRNodeV2::is_leaf() const {
    return op == IROpV2::INPUT || 
           op == IROpV2::PARAM || 
           op == IROpV2::WEIGHT;
}

inline bool IRNodeV2::is_param() const {
    return op == IROpV2::PARAM || op == IROpV2::WEIGHT;
}

inline std::string IRNodeV2::op_name() const {
    switch(op) {
        case IROpV2::INPUT: return "input";
        case IROpV2::PARAM: return "param";
        case IROpV2::WEIGHT: return "weight";
        case IROpV2::MATMUL: return "matmul";
        case IROpV2::ADD: return "add";
        case IROpV2::MUL: return "mul";
        case IROpV2::SUB: return "sub";
        case IROpV2::BIAS_ADD: return "bias_add";
        case IROpV2::RELU: return "relu";
        case IROpV2::SILU: return "silu";
        case IROpV2::GELU: return "gelu";
        case IROpV2::SOFTMAX: return "softmax";
        case IROpV2::LAYERNORM: return "layernorm";
        case IROpV2::RMSNORM: return "rmsnorm";
        case IROpV2::ATTENTION_QKV: return "attention_qkv";
        case IROpV2::FUSED_GEMM_BIAS_RELU: return "fused_gemm_bias_relu";
        case IROpV2::FUSED_ATTENTION_BLOCK: return "fused_attention_block";
        case IROpV2::FUSED_MLP_BLOCK: return "fused_mlp_block";
        case IROpV2::FUSED_ELEMENTWISE_BLOCK: return "fused_elementwise_block";
        default: return "unknown";
    }
}

inline IRNodeV2::IRNodeV2(size_t i, const std::string& n, IROpV2 o)
    : id(i), name(n), op(o) {}

inline IRNodeV2::IRNodeV2(size_t i, const std::string& n, IROpV2 o, const IRShape& s)
    : id(i), name(n), op(o), shape(s) {}

inline void IRNodeV2::add_operand(IRValue* val, size_t index) {
    operands.push_back(IRUse(val, index));
}

inline void IRNodeV2::add_result(IRValue* val) {
    results.push_back(val);
}

inline std::string IRNodeV2::to_string() const {
    return name + " = " + op_name() + "(" + std::to_string(shape.dims[0]) + "," + std::to_string(shape.dims.size() > 1 ? shape.dims[1] : 0) + ")";
}

}
