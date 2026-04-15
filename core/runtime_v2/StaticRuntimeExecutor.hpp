#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include "../compiler_v4/GraphCompilerV4.hpp"

namespace aresml {

class StaticRuntimeExecutor {
public:
    StaticRuntimeExecutor() : use_simd_(true), use_fusion_(true) {}
    
    explicit StaticRuntimeExecutor(bool use_simd, bool use_fusion)
        : use_simd_(use_simd), use_fusion_(use_fusion) {}
    
    void execute(StaticExecutionPlan& plan, float* buffer_pool) {
        if (plan.empty()) return;
        
        for (size_t i = 0; i < plan.nodes.size(); ++i) {
            auto& node = plan.nodes[i];
            
            float* input1 = buffer_pool + node.input_buffer_ids[0];
            float* input2 = buffer_pool + node.input_buffer_ids[1];
            float* output = buffer_pool + node.output_buffer_ids[0];
            
            execute_kernel(node.kernel_name, input1, input2, output, node.is_fused);
        }
    }
    
    void execute_fused(StaticExecutionPlan& plan, float* buffer_pool) {
        if (plan.empty()) return;
        
        size_t i = 0;
        while (i < plan.nodes.size()) {
            auto& node = plan.nodes[i];
            
            if (node.is_fused && use_fusion_) {
                float* input1 = buffer_pool + node.input_buffer_ids[0];
                float* input2 = buffer_pool + node.input_buffer_ids[1];
                float* output = buffer_pool + node.output_buffer_ids[0];
                
                execute_fused_kernel(node.kernel_name, input1, input2, output);
                
                i += 2;
            } else {
                float* input1 = buffer_pool + node.input_buffer_ids[0];
                float* input2 = buffer_pool + node.input_buffer_ids[1];
                float* output = buffer_pool + node.output_buffer_ids[0];
                
                execute_kernel(node.kernel_name, input1, input2, output, false);
                i++;
            }
        }
    }
    
    void set_simd_enabled(bool enabled) { use_simd_ = enabled; }
    void set_fusion_enabled(bool enabled) { use_fusion_ = enabled; }
    
private:
    bool use_simd_;
    bool use_fusion_;
    
    void execute_kernel(const std::string& name, float* a, float* b, float* c, bool fused) {
        if (name == "gemm") {
            execute_gemm(c, a, b, 32, 32, 32);
        } else if (name == "add") {
            execute_add(c, a, b, 1024);
        } else if (name == "mul") {
            execute_mul(c, a, b, 1024);
        } else if (name == "relu") {
            execute_relu(c, a, 1024);
        } else if (name == "gelu") {
            execute_gelu(c, a, 1024);
        } else if (name == "fused_gemm_bias_relu") {
            execute_fused_gemm_bias_relu(c, a, b, 32, 32, 32);
        }
    }
    
    void execute_fused_kernel(const std::string& name, float* a, float* b, float* c) {
        execute_kernel(name, a, b, c, true);
    }
    
    void execute_gemm(float* c, const float* a, const float* b, size_t m, size_t n, size_t k) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t p = 0; p < k; ++p) {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
    
    void execute_add(float* out, const float* a, const float* b, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            out[i] = a[i] + b[i];
        }
    }
    
    void execute_mul(float* out, const float* a, const float* b, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            out[i] = a[i] * b[i];
        }
    }
    
    void execute_relu(float* out, const float* in, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            out[i] = in[i] > 0.0f ? in[i] : 0.0f;
        }
    }
    
    void execute_gelu(float* out, const float* in, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            float x = in[i];
            out[i] = 0.5f * x * (1.0f + tanhf(0.797885f * x * (1.0f + 0.0331653f * x * x)));
        }
    }
    
    void execute_fused_gemm_bias_relu(float* out, const float* a, const float* b, 
                                       size_t m, size_t n, size_t k) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t p = 0; p < k; ++p) {
                    sum += a[i * k + p] * b[p * n + j];
                }
                out[i * n + j] = sum > 0.0f ? sum : 0.0f;
            }
        }
    }
};

}
