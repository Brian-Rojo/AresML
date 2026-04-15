#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include "../execution_v3/ExecutionPlan.hpp"

namespace aresml {

class PlanExecutor {
public:
    PlanExecutor() : num_threads_(4), verbose_(false) {}
    
    explicit PlanExecutor(size_t threads) : num_threads_(threads), verbose_(false) {}
    
    void execute(const ExecutionPlan& plan, float* input_buffer, float* output_buffer) {
        if (verbose_) {
            printf("[PlanExecutor] Starting execution of %zu tasks\n", plan.task_count());
        }
        
        execute_linear(plan, input_buffer, output_buffer);
    }
    
    void execute_linear(const ExecutionPlan& plan, float* input_buffer, float* output_buffer) {
        float* current_buffer = input_buffer;
        
        for (size_t i = 0; i < plan.tasks.size(); ++i) {
            const auto& task = plan.tasks[i];
            
            execute_task(task, current_buffer, output_buffer);
            
            if (verbose_) {
                printf("[PlanExecutor] Executed task %zu: %s\n", i, task.kernel_name.c_str());
            }
        }
    }
    
    void execute_parallel(const ExecutionPlan& plan, float* input_buffer, float* output_buffer) {
        std::atomic<size_t> next_task(0);
        std::vector<std::thread> threads;
        
        for (size_t t = 0; t < num_threads_; ++t) {
            threads.emplace_back([this, &plan, &next_task, input_buffer, output_buffer]() {
                while (true) {
                    size_t i = next_task.fetch_add(1);
                    if (i >= plan.tasks.size()) break;
                    
                    const auto& task = plan.tasks[i];
                    execute_task(task, input_buffer, output_buffer);
                }
            });
        }
        
        for (auto& th : threads) {
            th.join();
        }
    }
    
    void set_num_threads(size_t n) { num_threads_ = n; }
    void set_verbose(bool v) { verbose_ = v; }
    
private:
    size_t num_threads_;
    bool verbose_;
    
    void execute_task(const KernelTask& task, float* input, float* output) {
        switch(task.op) {
            case IROpV3::GEMM:
            case IROpV3::GEMM_BIAS_RELU:
                execute_gemm(task, input, output);
                break;
            case IROpV3::ELEMENTWISE_ADD:
                execute_add(task, input, output);
                break;
            case IROpV3::ELEMENTWISE_MUL:
                execute_mul(task, input, output);
                break;
            case IROpV3::ELEMENTWISE_RELU:
                execute_relu(task, input, output);
                break;
            case IROpV3::ELEMENTWISE_GELU:
                execute_gelu(task, input, output);
                break;
            default:
                break;
        }
    }
    
    void execute_gemm(const KernelTask& task, float* input, float* output) {
        size_t m = task.spec.m;
        size_t n = task.spec.n;
        size_t k = task.spec.k;
        
        if (task.spec.use_blocking) {
            execute_gemm_blocked(input, output, m, n, k,
                                task.spec.block_m, task.spec.block_n, task.spec.block_k);
        } else {
            execute_gemm_naive(input, output, m, n, k);
        }
    }
    
    void execute_gemm_naive(float* a, float* c, size_t m, size_t n, size_t k) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (size_t p = 0; p < k; ++p) {
                    sum += a[i * k + p] * a[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
    
    void execute_gemm_blocked(float* a, float* c, size_t m, size_t n, size_t k,
                              size_t bm, size_t bn, size_t bk) {
        for (size_t ii = 0; ii < m; ii += bm) {
            for (size_t jj = 0; jj < n; jj += bn) {
                for (size_t kk = 0; kk < k; kk += bk) {
                    size_t mi = std::min(bm, m - ii);
                    size_t nj = std::min(bn, n - jj);
                    size_t kp = std::min(bk, k - kk);
                    
                    for (size_t i = 0; i < mi; ++i) {
                        for (size_t j = 0; j < nj; ++j) {
                            float sum = 0.0f;
                            for (size_t p = 0; p < kp; ++p) {
                                sum += a[(ii + i) * k + kk + p] * a[(kk + p) * n + (jj + j)];
                            }
                            c[(ii + i) * n + (jj + j)] += sum;
                        }
                    }
                }
            }
        }
    }
    
    void execute_add(const KernelTask& task, float* input, float* output) {
        size_t n = task.spec.n;
        for (size_t i = 0; i < n; ++i) {
            output[i] = input[i] + input[i];
        }
    }
    
    void execute_mul(const KernelTask& task, float* input, float* output) {
        size_t n = task.spec.n;
        for (size_t i = 0; i < n; ++i) {
            output[i] = input[i] * input[i];
        }
    }
    
    void execute_relu(const KernelTask& task, float* input, float* output) {
        size_t n = task.spec.n;
        for (size_t i = 0; i < n; ++i) {
            output[i] = input[i] > 0.0f ? input[i] : 0.0f;
        }
    }
    
    void execute_gelu(const KernelTask& task, float* input, float* output) {
        size_t n = task.spec.n;
        for (size_t i = 0; i < n; ++i) {
            float x = input[i];
            output[i] = 0.5f * x * (1.0f + tanhf(0.797885f * x * (1.0f + 0.0331653f * x * x)));
        }
    }
};

}
