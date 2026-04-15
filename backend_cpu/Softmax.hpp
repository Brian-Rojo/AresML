#pragma once

#include "../core/Tensor.hpp"
#include "../ops/Ops.hpp"
#include "../utils/Profiler.hpp"
#include <cmath>

namespace aresml {
namespace backend_cpu {

inline void softmax(const Tensor& x, Tensor& out, int axis = -1) {
    if (x.shape.n == 0) return;
    
    size_t ndim = x.shape.n;
    size_t axis_idx = (axis < 0) ? static_cast<int>(ndim) + axis : static_cast<size_t>(axis);
    if (axis_idx >= ndim) axis_idx = ndim - 1;
    
    if (ndim == 1 || (ndim == 2 && axis_idx == 1)) {
        size_t rows = (ndim == 1) ? 1 : x.shape[0];
        size_t cols = (ndim == 1) ? x.shape[0] : x.shape[1];
        
        for (size_t i = 0; i < rows; ++i) {
            float max_val = -1e38f;
            for (size_t j = 0; j < cols; ++j) {
                size_t idx = (ndim == 1) ? j : i * cols + j;
                max_val = std::max(max_val, x.data[x.offset + idx]);
            }
            
            float sum = 0.0f;
            for (size_t j = 0; j < cols; ++j) {
                size_t idx = (ndim == 1) ? j : i * cols + j;
                sum += std::exp(x.data[x.offset + idx] - max_val);
            }
            
            for (size_t j = 0; j < cols; ++j) {
                size_t idx = (ndim == 1) ? j : i * cols + j;
                out.data[out.offset + idx] = std::exp(x.data[x.offset + idx] - max_val) / (sum + EPSILON);
            }
        }
    } else if (ndim == 2 && axis_idx == 0) {
        size_t rows = x.shape[0];
        size_t cols = x.shape[1];
        
        for (size_t j = 0; j < cols; ++j) {
            float max_val = -1e38f;
            for (size_t i = 0; i < rows; ++i) {
                max_val = std::max(max_val, x.data[x.offset + i * cols + j]);
            }
            
            float sum = 0.0f;
            for (size_t i = 0; i < rows; ++i) {
                sum += std::exp(x.data[x.offset + i * cols + j] - max_val);
            }
            
            for (size_t i = 0; i < rows; ++i) {
                out.data[out.offset + i * cols + j] = std::exp(x.data[x.offset + i * cols + j] - max_val) / (sum + EPSILON);
            }
        }
    } else if (ndim == 3 && axis_idx == 2) {
        size_t batch = x.shape[0];
        size_t seq = x.shape[1];
        size_t vocab = x.shape[2];
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < seq; ++s) {
                float max_val = -1e38f;
                for (size_t v = 0; v < vocab; ++v) {
                    size_t idx = b * seq * vocab + s * vocab + v;
                    max_val = std::max(max_val, x.data[x.offset + idx]);
                }
                
                float sum = 0.0f;
                for (size_t v = 0; v < vocab; ++v) {
                    size_t idx = b * seq * vocab + s * vocab + v;
                    sum += std::exp(x.data[x.offset + idx] - max_val);
                }
                
                for (size_t v = 0; v < vocab; ++v) {
                    size_t idx = b * seq * vocab + s * vocab + v;
                    out.data[out.offset + idx] = std::exp(x.data[x.offset + idx] - max_val) / (sum + EPSILON);
                }
            }
        }
    }
}

inline Tensor softmax(const Tensor& x, int axis = -1) {
    PROFILE_SCOPE("backend::softmax");
    Tensor out(x.shape, x.requires_grad);
    softmax(x, out, axis);

    if (x.requires_grad) {
        out.op = std::make_unique<ops::SoftmaxOp>(const_cast<Tensor*>(&x), axis);
        out.inputs.clear();
        out.inputs.push_back(const_cast<Tensor*>(&x));
    }

    return out;
}

inline void log_softmax(const Tensor& x, Tensor& out, int axis = -1) {
    // Direct computation for numerical stability
    // log_softmax(x_i) = x_i - max(x) - log(sum_j(exp(x_j - max(x))))
    if (x.shape.n == 0) return;
    
    size_t ndim = x.shape.n;
    size_t axis_idx = (axis < 0) ? static_cast<int>(ndim) + axis : static_cast<size_t>(axis);
    if (axis_idx >= ndim) axis_idx = ndim - 1;
    
    if (ndim == 1 || (ndim == 2 && axis_idx == 1)) {
        size_t rows = (ndim == 1) ? 1 : x.shape[0];
        size_t cols = (ndim == 1) ? x.shape[0] : x.shape[1];
        
        for (size_t i = 0; i < rows; ++i) {
            float max_val = -1e38f;
            for (size_t j = 0; j < cols; ++j) {
                size_t idx = (ndim == 1) ? j : i * cols + j;
                max_val = std::max(max_val, x.data[x.offset + idx]);
            }
            
            float sum_exp = 0.0f;
            for (size_t j = 0; j < cols; ++j) {
                size_t idx = (ndim == 1) ? j : i * cols + j;
                sum_exp += std::exp(x.data[x.offset + idx] - max_val);
            }
            float log_sum_exp = std::log(sum_exp + EPSILON);
            
            for (size_t j = 0; j < cols; ++j) {
                size_t idx = (ndim == 1) ? j : i * cols + j;
                out.data[out.offset + idx] = x.data[x.offset + idx] - max_val - log_sum_exp;
            }
        }
    } else if (ndim == 2 && axis_idx == 0) {
        size_t rows = x.shape[0];
        size_t cols = x.shape[1];
        
        for (size_t j = 0; j < cols; ++j) {
            float max_val = -1e38f;
            for (size_t i = 0; i < rows; ++i) {
                max_val = std::max(max_val, x.data[x.offset + i * cols + j]);
            }
            
            float sum_exp = 0.0f;
            for (size_t i = 0; i < rows; ++i) {
                sum_exp += std::exp(x.data[x.offset + i * cols + j] - max_val);
            }
            float log_sum_exp = std::log(sum_exp + EPSILON);
            
            for (size_t i = 0; i < rows; ++i) {
                out.data[out.offset + i * cols + j] = x.data[x.offset + i * cols + j] - max_val - log_sum_exp;
            }
        }
    } else if (ndim == 3 && axis_idx == 2) {
        size_t batch = x.shape[0];
        size_t seq = x.shape[1];
        size_t vocab = x.shape[2];
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < seq; ++s) {
                float max_val = -1e38f;
                for (size_t v = 0; v < vocab; ++v) {
                    size_t idx = b * seq * vocab + s * vocab + v;
                    max_val = std::max(max_val, x.data[x.offset + idx]);
                }
                
                float sum_exp = 0.0f;
                for (size_t v = 0; v < vocab; ++v) {
                    size_t idx = b * seq * vocab + s * vocab + v;
                    sum_exp += std::exp(x.data[x.offset + idx] - max_val);
                }
                float log_sum_exp = std::log(sum_exp + EPSILON);
                
                for (size_t v = 0; v < vocab; ++v) {
                    size_t idx = b * seq * vocab + s * vocab + v;
                    out.data[out.offset + idx] = x.data[x.offset + idx] - max_val - log_sum_exp;
                }
            }
        }
    }
}

inline Tensor log_softmax(const Tensor& x, int axis = -1) {
    PROFILE_SCOPE("backend::log_softmax");
    Tensor out(x.shape, x.requires_grad);
    log_softmax(x, out, axis);

    if (x.requires_grad) {
        out.op = std::make_unique<ops::LogSoftmaxOp>(const_cast<Tensor*>(&x), axis);
        out.inputs.clear();
        out.inputs.push_back(const_cast<Tensor*>(&x));
    }

    return out;
}

}
}
