#include <iostream>
#include <cmath>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "ops/Ops.hpp"

using namespace aresml;
using namespace aresml::ops;

int main() {
    std::cout << "=== AUTOGRAD MATH ENGINE TEST ===\n";
    
    int passed = 0, total = 0;
    
    get_engine().zero_grad();
    get_engine().set_debug(false);
    
    // TEST 1: add operation
    std::cout << "\nTEST 1: add operation (x + x)\n";
    ++total;
    {
        Tensor x({2, 4});
        x.fill(1.0f);
        x.set_requires_grad(true);
        
        Tensor y = add(x, x);
        Tensor loss = sum(y);
        
        backward(loss);
        
        if (x.grad && x.grad->data && std::abs(x.grad->data[0] - 2.0f) < 0.01f) {
            std::cout << "PASS: x.grad = " << x.grad->data[0] << " (expected 2)\n";
            ++passed;
        } else {
            std::cout << "FAIL: x.grad = " << (x.grad && x.grad->data ? x.grad->data[0] : 0) << "\n";
        }
    }
    get_engine().zero_grad();
    
    // TEST 2: mul operation (x * x)
    std::cout << "\nTEST 2: mul operation (x * x)\n";
    ++total;
    {
        Tensor x({2, 4});
        x.fill(2.0f);
        x.set_requires_grad(true);
        
        Tensor y = mul(x, x);  // x * x
        Tensor loss = sum(y);
        
        backward(loss);
        
        // d(x*x)/dx = 2x = 4
        if (x.grad && x.grad->data && std::abs(x.grad->data[0] - 4.0f) < 0.01f) {
            std::cout << "PASS: x.grad = " << x.grad->data[0] << " (expected 4)\n";
            ++passed;
        } else {
            std::cout << "FAIL: x.grad = " << (x.grad && x.grad->data ? x.grad->data[0] : 0) << "\n";
        }
    }
    get_engine().zero_grad();
    
    // TEST 3: mul with scalar (x * 2)
    std::cout << "\nTEST 3: mul scalar (x * 2)\n";
    ++total;
    {
        Tensor x({2, 4});
        x.fill(3.0f);
        x.set_requires_grad(true);
        
        Tensor two({2, 4});
        two.fill(2.0f);
        
        Tensor y = mul(x, two);
        Tensor loss = sum(y);
        
        backward(loss);
        
        // d(x*2)/dx = 2
        if (x.grad && x.grad->data && std::abs(x.grad->data[0] - 2.0f) < 0.01f) {
            std::cout << "PASS: x.grad = " << x.grad->data[0] << " (expected 2)\n";
            ++passed;
        } else {
            std::cout << "FAIL: x.grad = " << (x.grad && x.grad->data ? x.grad->data[0] : 0) << "\n";
        }
    }
    get_engine().zero_grad();
    
    // TEST 4: matmul operation
    std::cout << "\nTEST 4: matmul operation\n";
    ++total;
    {
        Tensor A({2, 3});
        A.fill(1.0f);
        A.set_requires_grad(true);
        
        Tensor B({3, 4});
        B.fill(1.0f);
        B.set_requires_grad(true);
        
        Tensor C = matmul(A, B);  // 2x3 @ 3x4 = 2x4
        Tensor loss = sum(C);
        
        backward(loss);
        
        // Each element of A contributes 4 (sum over B columns)
        if (A.grad && A.grad->data && std::abs(A.grad->data[0] - 4.0f) < 0.01f) {
            std::cout << "PASS: A.grad = " << A.grad->data[0] << " (expected 4)\n";
            ++passed;
        } else {
            std::cout << "FAIL: A.grad = " << (A.grad && A.grad->data ? A.grad->data[0] : 0) << "\n";
        }
    }
    get_engine().zero_grad();
    
    // TEST 5: combined (x * 2 + 3)
    std::cout << "\nTEST 5: combined (x * 2 + 3)\n";
    ++total;
    {
        Tensor x({2, 4});
        x.fill(1.0f);
        x.set_requires_grad(true);
        
        Tensor two({2, 4});
        two.fill(2.0f);
        
        Tensor y = mul(x, two);
        Tensor z = add(y, two);  // x*2 + 2
        Tensor loss = sum(z);
        
        backward(loss);
        
        // d/dx of (x*2 + 2) = 2
        if (x.grad && x.grad->data && std::abs(x.grad->data[0] - 2.0f) < 0.01f) {
            std::cout << "PASS: x.grad = " << x.grad->data[0] << " (expected 2)\n";
            ++passed;
        } else {
            std::cout << "FAIL: x.grad = " << (x.grad && x.grad->data ? x.grad->data[0] : 0) << "\n";
        }
    }
    get_engine().zero_grad();
    
    // TEST 6: relu
    std::cout << "\nTEST 6: relu operation\n";
    ++total;
    {
        Tensor x({4});
        x.data[0] = -1.0f;
        x.data[1] = 0.5f;
        x.data[2] = -0.5f;
        x.data[3] = 1.0f;
        x.set_requires_grad(true);
        
        Tensor y = relu(x);
        Tensor loss = sum(y);
        
        backward(loss);
        
        // grad only flows where x > 0
        if (x.grad && x.grad->data) {
            bool pass = (x.grad->data[0] == 0.0f && x.grad->data[1] == 1.0f &&
                        x.grad->data[2] == 0.0f && x.grad->data[3] == 1.0f);
            if (pass) {
                std::cout << "PASS: relu gradients correct\n";
                ++passed;
            } else {
                std::cout << "FAIL: got " << x.grad->data[0] << "," << x.grad->data[1] 
                          << "," << x.grad->data[2] << "," << x.grad->data[3] << "\n";
            }
        } else {
            std::cout << "FAIL: no grad\n";
        }
    }
    
    std::cout << "\n=== RESULTS: " << passed << "/" << total << " ===\n";
    
    return passed == total ? 0 : 1;
}