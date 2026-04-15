#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#include "core/Tensor.hpp"
#include "core/Autograd.hpp"
#include "nn/Linear.hpp"
#include "loss/CrossEntropy.hpp"

using namespace aresml;

int main() {
    std::cout << "=== Test: View/Reshape/Transpose/Permute ===\n";
    
    get_engine().zero_grad();
    
    // Test 1: view()
    std::cout << "--- Test 1: view() ---\n";
    {
        Tensor x({2, 4});
        x.fill(1.0f);
        x.set_requires_grad(true);
        get_engine().register_parameter(&x);
        
        Tensor y = x.view({4, 2});
        if (y.grad) y.grad->fill(1.0f);
        backward(y);
        
        if (x.grad && x.grad->data && x.grad->data[0] == 1.0f) {
            std::cout << "PASS\n";
        } else {
            std::cout << "FAIL\n";
        }
    }
    get_engine().zero_grad();
    
    // Test 2: reshape()
    std::cout << "--- Test 2: reshape() ---\n";
    {
        Tensor a({2, 4});
        a.fill(2.0f);
        a.set_requires_grad(true);
        get_engine().register_parameter(&a);
        
        Tensor b = a.reshape({4, 2});
        if (b.grad) b.grad->fill(1.0f);
        backward(b);
        
        if (a.grad && a.grad->data && a.grad->data[0] == 1.0f) {
            std::cout << "PASS\n";
        } else {
            std::cout << "FAIL\n";
        }
    }
    get_engine().zero_grad();
    
    // Test 3: transpose()
    std::cout << "--- Test 3: transpose() ---\n";
    {
        Tensor c({2, 3});
        c.fill(3.0f);
        c.set_requires_grad(true);
        get_engine().register_parameter(&c);
        
        Tensor d = c.transpose(0, 1);
        if (d.grad) d.grad->fill(1.0f);
        backward(d);
        
        if (c.grad && c.grad->data && c.grad->data[0] == 1.0f) {
            std::cout << "PASS\n";
        } else {
            std::cout << "FAIL\n";
        }
    }
    get_engine().zero_grad();
    
    // Test 4: permute()
    std::cout << "--- Test 4: permute() ---\n";
    {
        Tensor e({2, 3, 4, 5});
        e.fill(4.0f);
        e.set_requires_grad(true);
        get_engine().register_parameter(&e);
        
        Tensor f = e.permute({0, 2, 1, 3});
        if (f.grad) f.grad->fill(1.0f);
        backward(f);
        
        if (e.grad && e.grad->data && e.grad->data[0] == 1.0f) {
            std::cout << "PASS\n";
        } else {
            std::cout << "FAIL\n";
        }
    }
    get_engine().zero_grad();
    
    // Test 5: Linear + CE
    std::cout << "--- Test 5: Linear + CE ---\n";
    {
        nn::Linear linear(8, 16);
        
        Tensor input({2, 8});
        input.fill(0.5f);
        input.set_requires_grad(true);
        
        Tensor output = linear.forward(input);
        
        Tensor targets({2});
        targets.data[0] = 0;
        targets.data[1] = 1;
        
        loss::CrossEntropyLoss loss_fn(true);
        Tensor loss5 = loss_fn.forward(output, targets);
        
        backward(loss5);
        
        if (linear.weight.grad && linear.weight.grad->data && linear.weight.grad->data[0] != 0.0f) {
            std::cout << "PASS\n";
        } else {
            std::cout << "FAIL\n";
        }
    }
    
    std::cout << "\n=== Done ===\n";
    return 0;
}