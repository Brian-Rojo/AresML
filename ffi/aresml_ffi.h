#ifndef ARESML_FFI_H
#define ARESML_FFI_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ARESML_SUCCESS = 0,
    ARESML_ERR_NULLPTR = 1,
    ARESML_ERR_INVALID_SHAPE = 2,
    ARESML_ERR_OUT_OF_MEMORY = 3,
    ARESML_ERR_DIMENSION_MISMATCH = 4,
    ARESML_ERR_INVALID_OPERATION = 5,
    ARESML_ERR_GRAPH_ERROR = 6,
    ARESML_ERR_BACKWARD_FAILED = 7,
    ARESML_ERR_OPTIMIZER_ERROR = 8,
    ARESML_ERR_CUSTOM = 9
} aresml_error_t;

typedef void* AresmlGraphContext;
typedef void* AresmlTensor;
typedef void* AresmlLinear;
typedef void* AresmlSGD;
typedef void* AresmlAdam;

AresmlGraphContext aresml_graph_context_create(void);
void aresml_graph_context_destroy(AresmlGraphContext ctx);
aresml_error_t aresml_graph_register_leaf(AresmlGraphContext ctx, AresmlTensor tensor);
aresml_error_t aresml_graph_zero_grad(AresmlGraphContext ctx);
aresml_error_t aresml_graph_backward(AresmlGraphContext ctx, AresmlTensor loss);

AresmlTensor aresml_tensor_create(const int64_t* shape, int32_t ndim);
AresmlTensor aresml_tensor_randn(const int64_t* shape, int32_t ndim);
AresmlTensor aresml_tensor_zeros(const int64_t* shape, int32_t ndim);
AresmlTensor aresml_tensor_clone(AresmlTensor tensor);
void aresml_tensor_free(AresmlTensor tensor);
int64_t* aresml_tensor_get_shape(AresmlTensor tensor, int32_t* out_ndim);
int64_t aresml_tensor_size(AresmlTensor tensor);
const float* aresml_tensor_data(AresmlTensor tensor);
aresml_error_t aresml_tensor_set_requires_grad(AresmlTensor tensor, bool value);
bool aresml_tensor_requires_grad(AresmlTensor tensor);
bool aresml_tensor_has_nan(AresmlTensor tensor);

AresmlTensor aresml_add(AresmlTensor a, AresmlTensor b);
AresmlTensor aresml_mul(AresmlTensor a, AresmlTensor b);
AresmlTensor aresml_matmul(AresmlTensor a, AresmlTensor b);
AresmlTensor aresml_relu(AresmlTensor x);
AresmlTensor aresml_softmax(AresmlTensor x, int32_t axis);
AresmlTensor aresml_log_softmax(AresmlTensor x, int32_t axis);
AresmlTensor aresml_sum(AresmlTensor x);
AresmlTensor aresml_mean(AresmlTensor x);

AresmlLinear aresml_linear_create(int32_t in_features, int32_t out_features, bool bias);
AresmlTensor aresml_linear_forward(AresmlLinear layer, AresmlTensor input);
const float* aresml_linear_get_weight_data(AresmlLinear layer);
const float* aresml_linear_get_bias_data(AresmlLinear layer);
void aresml_linear_free(AresmlLinear layer);

AresmlTensor aresml_mse_loss(AresmlTensor pred, AresmlTensor target);
AresmlTensor aresml_cross_entropy(AresmlTensor logits, AresmlTensor targets);

const char* aresml_get_last_error(void);
void aresml_clear_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
