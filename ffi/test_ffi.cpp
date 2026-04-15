#include "aresml_ffi.h"
#include <cstdio>

int main() {
    printf("=== FFI Test ===\n");
    
    // Create context
    auto ctx = aresml_graph_context_create();
    printf("Context created: %p\n", (void*)ctx);
    
    // Create tensor
    int64_t shape[2] = {4, 8};
    auto t = aresml_tensor_randn(shape, 2);
    printf("Tensor created: %p\n", (void*)t);
    
    // Get data
    auto data = aresml_tensor_data(t);
    if (data) {
        printf("First value: %f\n", data[0]);
    }
    
    // Create linear
    auto linear = aresml_linear_create(8, 16, true);
    printf("Linear created: %p\n", (void*)linear);
    
    // Forward
    auto out = aresml_linear_forward(linear, t);
    printf("Output: %p\n", (void*)out);
    
    // Get output size
    auto out_size = aresml_tensor_size(out);
    printf("Output size: %ld\n", (long)out_size);
    
    // Free
    aresml_tensor_free(t);
    aresml_tensor_free(out);
    aresml_linear_free(linear);
    aresml_graph_context_destroy(ctx);
    
    printf("PASS\n");
    return 0;
}