#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace aresml {

enum class DType {
    float32,
    float16,
    int32,
    int64
};

enum class Device {
    cpu,
    cuda
};

enum class ShapeErrorType {
    MATMUL_MISMATCH,
    BROADCAST_ERROR,
    ATTENTION_MISMATCH,
    INVALID_DIMENSION,
    SIZE_MISMATCH
};

struct ShapeError {
    ShapeErrorType type;
    std::string message;
    
    ShapeError(ShapeErrorType t, const std::string& msg) 
        : type(t), message(msg) {}
    
    static ShapeError matmul_mismatch(size_t a_cols, size_t b_rows) {
        std::ostringstream ss;
        ss << "Matmul shape mismatch: A.cols=" << a_cols << " B.rows=" << b_rows;
        return ShapeError(ShapeErrorType::MATMUL_MISMATCH, ss.str());
    }
    
    static ShapeError attention_mismatch(const std::string& what) {
        std::ostringstream ss;
        ss << "Attention shape mismatch: " << what;
        return ShapeError(ShapeErrorType::ATTENTION_MISMATCH, ss.str());
    }
    
    static ShapeError broadcast_error(size_t a, size_t b) {
        std::ostringstream ss;
        ss << "Broadcast error: " << a << " vs " << b << " (not compatible)";
        return ShapeError(ShapeErrorType::BROADCAST_ERROR, ss.str());
    }
    
    static ShapeError size_mismatch(const std::string& op, size_t expected, size_t got) {
        std::ostringstream ss;
        ss << op << ": expected size " << expected << ", got " << got;
        return ShapeError(ShapeErrorType::SIZE_MISMATCH, ss.str());
    }
    
    void print() const {
        std::cerr << "[SHAPE ERROR] " << message << "\n";
    }
};

class TensorShape {
public:
    std::vector<size_t> dims;
    
    TensorShape() = default;
    
    TensorShape(const std::initializer_list<size_t>& init) : dims(init) {}
    
    TensorShape(const std::vector<size_t>& d) : dims(d) {}
    
    size_t operator[](size_t i) const {
        return i < dims.size() ? dims[i] : 1;
    }
    
    size_t& operator[](size_t i) {
        return dims[i];
    }
    
    size_t size() const {
        size_t s = 1;
        for (size_t d : dims) s *= d;
        return s;
    }
    
    size_t ndim() const {
        return dims.size();
    }
    
    bool operator==(const TensorShape& other) const {
        return dims == other.dims;
    }
    
    bool operator!=(const TensorShape& other) const {
        return dims != other.dims;
    }
    
    bool is_broadcastable_to(const TensorShape& target) const {
        size_t i = ndim();
        size_t j = target.ndim();
        
        while (i > 0 && j > 0) {
            size_t a = (i > 0) ? (*this)[i-1] : 1;
            size_t b = target[j-1];
            
            if (a != b && a != 1 && b != 1) {
                return false;
            }
            --i;
            --j;
        }
        return true;
    }
    
    TensorShape broadcast(const TensorShape& other) const {
        if (!is_broadcastable_to(other)) {
            throw std::runtime_error("Cannot broadcast: shapes incompatible");
        }
        
        size_t max_ndim = std::max(ndim(), other.ndim());
        std::vector<size_t> result(max_ndim, 1);
        
        for (size_t i = 0; i < max_ndim; ++i) {
            size_t a = (i < ndim()) ? dims[ndim() - 1 - i] : 1;
            size_t b = (i < other.ndim()) ? other.dims[other.ndim() - 1 - i] : 1;
            result[max_ndim - 1 - i] = std::max(a, b);
        }
        
        return TensorShape(result);
    }
    
    std::string to_string() const {
        std::ostringstream ss;
        ss << "(";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) ss << ",";
            ss << dims[i];
        }
        ss << ")";
        return ss.str();
    }
    
    bool is_contiguous() const {
        if (dims.empty()) return true;
        size_t expected = 1;
        for (size_t i = dims.size(); i > 0; --i) {
            if (dims[i-1] != expected) return false;
            expected *= dims[i-1];
        }
        return true;
    }
};

bool is_broadcast_compatible(const TensorShape& a, const TensorShape& b) {
    return a.is_broadcastable_to(b) || b.is_broadcastable_to(a);
}

TensorShape compute_broadcast(const TensorShape& a, const TensorShape& b) {
    if (!is_broadcast_compatible(a, b)) {
        throw std::runtime_error("Incompatible shapes for broadcast");
    }
    return a.is_broadcastable_to(b) ? a.broadcast(b) : b.broadcast(a);
}

class ShapeChecker {
public:
    static bool strict_mode;
    static bool debug_mode;
    
    static void enable_strict(bool enable) {
        strict_mode = enable;
        if (strict_mode) {
            std::cout << "[SHAPE] Strict mode enabled\n";
        }
    }
    
    static void enable_debug(bool enable) {
        debug_mode = enable;
    }
    
    static void validate_matmul(const TensorShape& a, const TensorShape& b) {
        if (a.ndim() < 2 || b.ndim() < 2) {
            throw ShapeError(ShapeErrorType::MATMUL_MISMATCH, 
                "Matmul requires at least 2D tensors");
        }
        
        if (a.ndim() != b.ndim()) {
            std::ostringstream ss;
            ss << "Matmul dimension mismatch: " << a.ndim() << "D vs " << b.ndim() << "D";
            throw ShapeError(ShapeErrorType::MATMUL_MISMATCH, ss.str());
        }
        
        if (a[1] != b[0]) {
            if (strict_mode) {
                throw ShapeError::matmul_mismatch(a[1], b[0]);
            }
        }
        
        if (debug_mode) {
            std::cout << "[SHAPE TRACE] Matmul: " << a.to_string() 
                      << " x " << b.to_string() << " -> " 
                      << TensorShape({a[0], b[1]}).to_string() << "\n";
        }
    }
    
    static void validate_add(const TensorShape& a, const TensorShape& b) {
        if (!is_broadcast_compatible(a, b)) {
            throw ShapeError::broadcast_error(a.size(), b.size());
        }
        
        if (strict_mode && a.size() != b.size() && !a.is_broadcastable_to(b)) {
            throw ShapeError::broadcast_error(a.size(), b.size());
        }
        
        if (debug_mode) {
            std::cout << "[SHAPE TRACE] Add: " << a.to_string() 
                      << " + " << b.to_string() << " -> "
                      << compute_broadcast(a, b).to_string() << "\n";
        }
    }
    
    static TensorShape infer_matmul(const TensorShape& a, const TensorShape& b) {
        validate_matmul(a, b);
        return TensorShape({a[0], b[1]});
    }
    
    static TensorShape infer_add(const TensorShape& a, const TensorShape& b) {
        validate_add(a, b);
        return is_broadcast_compatible(a, b) ? compute_broadcast(a, b) : a;
    }
    
    static void validate_attention(const TensorShape& Q, const TensorShape& K, const TensorShape& V) {
        if (Q.ndim() != 4 || K.ndim() != 4 || V.ndim() != 4) {
            throw ShapeError::attention_mismatch("expected 4D tensors (B,H,S,D)");
        }
        
        if (Q[0] != K[0] || Q[0] != V[0]) {
            throw ShapeError::attention_mismatch("batch mismatch");
        }
        
        if (Q[1] != K[1] || Q[1] != V[1]) {
            throw ShapeError::attention_mismatch("num_heads mismatch");
        }
        
        if (Q[2] != K[2] || Q[2] != V[2]) {
            throw ShapeError::attention_mismatch("seq_len mismatch");
        }
        
        if (Q[3] != K[3] || Q[3] != V[3]) {
            throw ShapeError::attention_mismatch("head_dim mismatch");
        }
        
        if (Q[2] != K[2]) {
            throw ShapeError::attention_mismatch("Q.seq != K.seq");
        }
        
        if (debug_mode) {
            std::cout << "[SHAPE TRACE] Attention validated: Q" << Q.to_string() 
                      << " K" << K.to_string() << " V" << V.to_string() << "\n";
        }
    }
    
    static TensorShape infer_attention(const TensorShape& Q, const TensorShape& K, const TensorShape& V) {
        validate_attention(Q, K, V);
        return Q;
    }
    
    static void validate_size(const TensorShape& a, size_t expected) {
        if (a.size() != expected) {
            throw ShapeError::size_mismatch("size", expected, a.size());
        }
    }
};

bool ShapeChecker::strict_mode = false;
bool ShapeChecker::debug_mode = false;

inline void set_shape_debug(bool enable) {
    ShapeChecker::enable_debug(enable);
}

}