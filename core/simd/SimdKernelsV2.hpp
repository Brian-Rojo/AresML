#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <immintrin.h>

namespace aresml {

namespace simd {

constexpr size_t FLOAT32_WIDTH = 8;
constexpr size_t CACHE_LINE = 64;

struct Float8 {
    __m256 vec;
    
    Float8() : vec(_mm256_setzero_ps()) {}
    Float8(__m256 v) : vec(v) {}
    Float8(float v) : vec(_mm256_set1_ps(v)) {}
    
    static Float8 zero() { return Float8(_mm256_setzero_ps()); }
    static Float8 load(const float* p) { return Float8(_mm256_loadu_ps(p)); }
    static Float8 load_aligned(const float* p) { return Float8(_mm256_load_ps(p)); }
    
    void store(float* p) const { _mm256_storeu_ps(p, vec); }
    void store_aligned(float* p) const { _mm256_store_ps(p, vec); }
    
    Float8 operator+(const Float8& other) const { return Float8(_mm256_add_ps(vec, other.vec)); }
    Float8 operator-(const Float8& other) const { return Float8(_mm256_sub_ps(vec, other.vec)); }
    Float8 operator*(const Float8& other) const { return Float8(_mm256_mul_ps(vec, other.vec)); }
    Float8 operator/(const Float8& other) const { return Float8(_mm256_div_ps(vec, other.vec)); }
    
    Float8& operator+=(const Float8& other) { vec = _mm256_add_ps(vec, other.vec); return *this; }
    Float8& operator*=(const Float8& other) { vec = _mm256_mul_ps(vec, other.vec); return *this; }
    
    Float8 relu() const {
        return Float8(_mm256_max_ps(vec, _mm256_setzero_ps()));
    }
    
    Float8 gelu() const {
        float tmp[8];
        _mm256_storeu_ps(tmp, vec);
        for (int i = 0; i < 8; ++i) {
            tmp[i] = 0.5f * tmp[i] * (1.0f + tanhf(0.797885f * tmp[i] * (1.0f + 0.0331653f * tmp[i] * tmp[i])));
        }
        return Float8(_mm256_loadu_ps(tmp));
    }
};

inline void add_f32(float* out, const float* a, const float* b, size_t n) {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        Float8 va = Float8::load(a + i);
        Float8 vb = Float8::load(b + i);
        (va + vb).store(out + i);
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

inline void mul_f32(float* out, const float* a, const float* b, size_t n) {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        Float8 va = Float8::load(a + i);
        Float8 vb = Float8::load(b + i);
        (va * vb).store(out + i);
    }
    for (; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

inline void relu_f32(float* out, const float* in, size_t n) {
    size_t i = 0;
    __m256 zero = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        Float8 vin = Float8::load(in + i);
        vin.relu().store(out + i);
    }
    for (; i < n; ++i) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}

inline void gelu_f32(float* out, const float* in, size_t n) {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        Float8 vin = Float8::load(in + i);
        vin.gelu().store(out + i);
    }
    for (; i < n; ++i) {
        float x = in[i];
        out[i] = 0.5f * x * (1.0f + tanhf(0.797885f * x * (1.0f + 0.0331653f * x * x)));
    }
}

inline float sum_f32(const float* data, size_t n) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    
    for (; i + 7 < n; i += 8) {
        sum = _mm256_add_ps(sum, _mm256_loadu_ps(data + i));
    }
    
    float result[8];
    _mm256_storeu_ps(result, sum);
    float total = result[0] + result[1] + result[2] + result[3] + 
                  result[4] + result[5] + result[6] + result[7];
    
    for (; i < n; ++i) {
        total += data[i];
    }
    
    return total;
}

inline void scale_f32(float* data, float scale, size_t n) {
    __m256 s = _mm256_set1_ps(scale);
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(data + i, _mm256_mul_ps(_mm256_loadu_ps(data + i), s));
    }
    for (; i < n; ++i) {
        data[i] *= scale;
    }
}

inline void fill_f32(float* data, float value, size_t n) {
    __m256 v = _mm256_set1_ps(value);
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(data + i, v);
    }
    for (; i < n; ++i) {
        data[i] = value;
    }
}

inline void copy_f32(float* dest, const float* src, size_t n) {
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(dest + i, _mm256_loadu_ps(src + i));
    }
    for (; i < n; ++i) {
        dest[i] = src[i];
    }
}

inline void matmul_blocked(float* out, const float* a, const float* b,
                          size_t m, size_t n, size_t k,
                          size_t block_m = 64, size_t block_n = 64, size_t block_k = 32) {
    for (size_t ii = 0; ii < m; ii += block_m) {
        for (size_t jj = 0; jj < n; jj += block_n) {
            for (size_t kk = 0; kk < k; kk += block_k) {
                size_t m_limit = std::min(block_m, m - ii);
                size_t n_limit = std::min(block_n, n - jj);
                size_t k_limit = std::min(block_k, k - kk);
                
                for (size_t i = 0; i < m_limit; ++i) {
                    for (size_t j = 0; j < n_limit; ++j) {
                        float sum = 0.0f;
                        for (size_t kk2 = 0; kk2 < k_limit; ++kk2) {
                            sum += a[(ii + i) * k + kk + kk2] * b[(kk + kk2) * n + (jj + j)];
                        }
                        out[(ii + i) * n + (jj + j)] += sum;
                    }
                }
            }
        }
    }
}

}

using namespace simd;

}
