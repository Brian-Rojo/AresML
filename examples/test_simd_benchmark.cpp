#include <iostream>
#include <chrono>
#include <random>
#include <cmath>

#include "core/simd/Simd.hpp"
#include "core/threading/Parallel.hpp"

using namespace aresml;
using namespace simd;
using namespace threading;

struct Benchmark {
    double time_ms;
    double bandwidth_gb_s;
    
    Benchmark(double t, size_t bytes) : time_ms(t), bandwidth_gb_s(bytes / 1e9 / (t / 1000.0)) {}
};

template<typename Func>
Benchmark measure(const char* name, size_t n, Func fn) {
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double bytes = n * sizeof(float) * 3; // read + read + write
    std::cout << name << ": " << ms << " ms (" << (bytes / 1e9 / (ms/1000.0)) << " GB/s)\n";
    return Benchmark(ms, bytes);
}

int main() {
    std::cout << "=== SIMD BENCHMARK ===\n";
    std::cout << "Hardware concurrency: " << get_num_threads() << " threads\n";
    std::cout << "SIMD width: " << simd_width() << " floats\n\n";
    
    // Large arrays
    size_t N = 10000000; // 10M elements
    std::cout << "Testing with " << N << " elements\n\n";
    
    std::vector<float> a(N), b(N), c(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (size_t i = 0; i < N; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }
    
    // Benchmark ADD
    std::cout << "--- ADD ---\n";
    measure("add (SIMD)", N, [&]() {
        add(a.data(), b.data(), c.data(), N);
    });
    
    // Benchmark MUL
    std::cout << "\n--- MUL ---\n";
    measure("mul (SIMD)", N, [&]() {
        mul(a.data(), b.data(), c.data(), N);
    });
    
    // Benchmark RELU (in-place)
    std::cout << "\n--- RELU ---\n";
    measure("relu (SIMD)", N, [&]() {
        c = a;
        relu(c.data(), N);
    });
    
    // Benchmark SUM
    std::cout << "\n--- SUM ---\n";
    float result = 0;
    measure("sum (SIMD)", N, [&]() {
        result = sum(a.data(), N);
    });
    std::cout << "Result: " << result << "\n";
    
    // Benchmark FILL
    std::cout << "\n--- FILL ---\n";
    measure("fill (SIMD)", N, [&]() {
        fill(c.data(), 3.14159f, N);
    });
    
    // Benchmark SCALE
    std::cout << "\n--- SCALE ---\n";
    measure("scale (SIMD)", N, [&]() {
        c = a;
        scale(c.data(), 2.0f, N);
    });
    
    // Benchmark COPY
    std::cout << "\n--- COPY ---\n";
    measure("copy (SIMD)", N, [&]() {
        copy(a.data(), c.data(), N);
    });
    
    std::cout << "\n=== DONE ===\n";
    
    return 0;
}