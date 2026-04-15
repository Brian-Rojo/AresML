#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __AVX2__
#include <immintrin.h>
#define USE_AVX2 1
#endif

namespace aresml {
namespace simd {

// ============================================================================
// SIMD utility functions
// ============================================================================

inline size_t simd_width() {
#ifdef __AVX2__
    return 8;
#else
    return 1;
#endif
}

inline size_t simd_align(size_t n) {
    size_t w = simd_width();
    return (n + w - 1) / w * w;
}

// ============================================================================
// Vectorized ADD
// ============================================================================

inline void add(const float* a, const float* b, float* out, size_t n) {
#ifdef USE_AVX2__
    size_t i = 0;
    __m256 ymm0, ymm1, ymm2;
    
    // Process 8 elements at a time
    for (; i + 8 <= n; i += 8) {
        ymm0 = _mm256_loadu_ps(a + i);
        ymm1 = _mm256_loadu_ps(b + i);
        ymm2 = _mm256_add_ps(ymm0, ymm1);
        _mm256_storeu_ps(out + i, ymm2);
    }
    
    // Tail - scalar
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
#endif
}

// ============================================================================
// Vectorized MUL
// ============================================================================

inline void mul(const float* a, const float* b, float* out, size_t n) {
#ifdef USE_AVX2__
    size_t i = 0;
    __m256 ymm0, ymm1, ymm2;
    
    for (; i + 8 <= n; i += 8) {
        ymm0 = _mm256_loadu_ps(a + i);
        ymm1 = _mm256_loadu_ps(b + i);
        ymm2 = _mm256_mul_ps(ymm0, ymm1);
        _mm256_storeu_ps(out + i, ymm2);
    }
    
    for (; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
#endif
}

// ============================================================================
// Vectorized RELU (in-place)
// ============================================================================

inline void relu(float* data, size_t n) {
#ifdef USE_AVX2__
    size_t i = 0;
    __m256 ymm0, ymm_zero;
    ymm_zero = _mm256_setzero_ps();
    
    for (; i + 8 <= n; i += 8) {
        ymm0 = _mm256_loadu_ps(data + i);
        ymm0 = _mm256_max_ps(ymm0, ymm_zero);
        _mm256_storeu_ps(data + i, ymm0);
    }
    
    for (; i < n; ++i) {
        if (data[i] < 0) data[i] = 0;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        if (data[i] < 0) data[i] = 0;
    }
#endif
}

// ============================================================================
// Vectorized SUM (horizontal add)
// ============================================================================

inline float sum(const float* data, size_t n) {
#ifdef USE_AVX2__
    __m256 ymm_sum = _mm256_setzero_ps();
    size_t i = 0;
    
    // Accumulate 8 elements at a time
    for (; i + 8 <= n; i += 8) {
        __m256 ymm0 = _mm256_loadu_ps(data + i);
        ymm_sum = _mm256_add_ps(ymm_sum, ymm0);
    }
    
    // Reduce 256-bit to 128-bit
    __m128 ymm_low = _mm256_castps256_ps128(ymm_sum);
    __m128 ymm_high = _mm256_extractf128_ps(ymm_sum, 1);
    __m128 ymm_total = _mm_add_ps(ymm_low, ymm_high);
    
    // Reduce to scalar
    float result = 0;
    float tmp[4];
    _mm_storeu_ps(tmp, ymm_total);
    result = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    
    // Tail
    for (; i < n; ++i) {
        result += data[i];
    }
    
    return result;
#else
    float result = 0;
    for (size_t i = 0; i < n; ++i) {
        result += data[i];
    }
    return result;
#endif
}

// ============================================================================
// Vectorized FILL
// ============================================================================

inline void fill(float* data, float value, size_t n) {
#ifdef USE_AVX2__
    __m256 ymm_val = _mm256_set1_ps(value);
    size_t i = 0;
    
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(data + i, ymm_val);
    }
    
    for (; i < n; ++i) {
        data[i] = value;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        data[i] = value;
    }
#endif
}

// ============================================================================
// Vectorized SCALE (multiply by scalar)
// ============================================================================

inline void scale(float* data, float value, size_t n) {
#ifdef USE_AVX2__
    __m256 ymm_val = _mm256_set1_ps(value);
    size_t i = 0;
    
    for (; i + 8 <= n; i += 8) {
        __m256 ymm0 = _mm256_loadu_ps(data + i);
        ymm0 = _mm256_mul_ps(ymm0, ymm_val);
        _mm256_storeu_ps(data + i, ymm0);
    }
    
    for (; i < n; ++i) {
        data[i] *= value;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        data[i] *= value;
    }
#endif
}

// ============================================================================
// Vectorized COPY
// ============================================================================

inline void copy(const float* src, float* dst, size_t n) {
#ifdef USE_AVX2__
    size_t i = 0;
    
    for (; i + 8 <= n; i += 8) {
        __m256 ymm0 = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, ymm0);
    }
    
    for (; i < n; ++i) {
        dst[i] = src[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
#endif
}

} // namespace simd
} // namespace aresml