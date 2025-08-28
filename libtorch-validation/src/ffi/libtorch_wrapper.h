#pragma once

#include <cstddef>
#include <cstdint>

// C interface for libtorch operations
extern "C" {
    // Tensor creation and management
    typedef struct LibtorchTensor LibtorchTensor;
    
    // Tensor creation
    LibtorchTensor* libtorch_tensor_new(const int64_t* shape, size_t ndim);
    LibtorchTensor* libtorch_tensor_zeros(const int64_t* shape, size_t ndim);
    LibtorchTensor* libtorch_tensor_ones(const int64_t* shape, size_t ndim);
    LibtorchTensor* libtorch_tensor_from_data(const float* data, const int64_t* shape, size_t ndim);
    
    // Tensor operations
    LibtorchTensor* libtorch_tensor_add_scalar(const LibtorchTensor* tensor, float scalar);
    LibtorchTensor* libtorch_tensor_add_tensor(const LibtorchTensor* a, const LibtorchTensor* b);
    LibtorchTensor* libtorch_tensor_sub_scalar(const LibtorchTensor* tensor, float scalar);
    LibtorchTensor* libtorch_tensor_sub_tensor(const LibtorchTensor* a, const LibtorchTensor* b);
    LibtorchTensor* libtorch_tensor_mul_scalar(const LibtorchTensor* tensor, float scalar);
    LibtorchTensor* libtorch_tensor_mul_tensor(const LibtorchTensor* a, const LibtorchTensor* b);
    LibtorchTensor* libtorch_tensor_div_scalar(const LibtorchTensor* tensor, float scalar);
    LibtorchTensor* libtorch_tensor_div_tensor(const LibtorchTensor* a, const LibtorchTensor* b);
    LibtorchTensor* libtorch_tensor_matmul(const LibtorchTensor* a, const LibtorchTensor* b);
    LibtorchTensor* libtorch_tensor_exp(const LibtorchTensor* a);
    LibtorchTensor* libtorch_tensor_sqrt(const LibtorchTensor* a);
    LibtorchTensor* libtorch_tensor_log(const LibtorchTensor* a);
    LibtorchTensor* libtorch_tensor_pow_scalar(const LibtorchTensor* a, float exponent);
    LibtorchTensor* libtorch_tensor_pow_tensor(const LibtorchTensor* a, const LibtorchTensor* b);
    LibtorchTensor* libtorch_tensor_softmax(const LibtorchTensor* a, int64_t dim);
    LibtorchTensor* libtorch_tensor_relu(const LibtorchTensor* a);
    LibtorchTensor* libtorch_tensor_tanh(const LibtorchTensor* a);
    LibtorchTensor* libtorch_tensor_sigmoid(const LibtorchTensor* a);
    LibtorchTensor* libtorch_tensor_leaky_relu(const LibtorchTensor* a, float negative_slope);
    LibtorchTensor* libtorch_tensor_sum(const LibtorchTensor* a);
    LibtorchTensor* libtorch_tensor_mean(const LibtorchTensor* a);
    LibtorchTensor* libtorch_tensor_std(const LibtorchTensor* a, bool unbiased);
    LibtorchTensor* libtorch_tensor_norm(const LibtorchTensor* a);
    LibtorchTensor* libtorch_tensor_var(const LibtorchTensor* a, bool unbiased);
    LibtorchTensor* libtorch_tensor_sum_dims(const LibtorchTensor* a, const int64_t* dims, size_t ndims, bool keepdim);
    LibtorchTensor* libtorch_tensor_mean_dims(const LibtorchTensor* a, const int64_t* dims, size_t ndims, bool keepdim);
    LibtorchTensor* libtorch_tensor_std_dims(const LibtorchTensor* a, const int64_t* dims, size_t ndims, bool keepdim, bool unbiased);
    LibtorchTensor* libtorch_tensor_norm_dims(const LibtorchTensor* a, const int64_t* dims, size_t ndims, bool keepdim);
    LibtorchTensor* libtorch_tensor_var_dims(const LibtorchTensor* a, const int64_t* dims, size_t ndims, bool keepdim, bool unbiased);
    LibtorchTensor* libtorch_tensor_min(const LibtorchTensor* a);
    LibtorchTensor* libtorch_tensor_min_dims(const LibtorchTensor* a, const int64_t* dims, size_t ndims, bool keepdim);
    LibtorchTensor* libtorch_tensor_max(const LibtorchTensor* a);
    LibtorchTensor* libtorch_tensor_max_dims(const LibtorchTensor* a, const int64_t* dims, size_t ndims, bool keepdim);
    LibtorchTensor* libtorch_tensor_argmin(const LibtorchTensor* a);
    LibtorchTensor* libtorch_tensor_argmin_dim(const LibtorchTensor* a, int64_t dim, bool keepdim);
    LibtorchTensor* libtorch_tensor_argmax(const LibtorchTensor* a);
    LibtorchTensor* libtorch_tensor_argmax_dim(const LibtorchTensor* a, int64_t dim, bool keepdim);
    LibtorchTensor* libtorch_tensor_cat(const LibtorchTensor** tensors, size_t count, int64_t dim);
    LibtorchTensor* libtorch_tensor_stack(const LibtorchTensor** tensors, size_t count, int64_t dim);
    LibtorchTensor* libtorch_tensor_view(const LibtorchTensor* tensor, const int64_t* shape, size_t ndim);
    LibtorchTensor* libtorch_tensor_permute(const LibtorchTensor* tensor, const int64_t* dims, size_t ndim);
    LibtorchTensor* libtorch_tensor_index_select(const LibtorchTensor* tensor, int64_t dim, const int64_t* indices, size_t nindices);
    LibtorchTensor* libtorch_tensor_gather(const LibtorchTensor* tensor, int64_t dim, const int64_t* index_data, const int64_t* index_shape, size_t index_ndim);
    LibtorchTensor* libtorch_tensor_masked_fill(const LibtorchTensor* tensor, const bool* mask, size_t numel, float value);
    LibtorchTensor* libtorch_tensor_select(const LibtorchTensor* tensor, int64_t dim, int64_t index);
    
    // Gradient operations (for autograd testing)
    LibtorchTensor* libtorch_tensor_require_grad(LibtorchTensor* tensor, bool requires_grad);
    void libtorch_tensor_backward(LibtorchTensor* tensor, LibtorchTensor* grad_output);
    LibtorchTensor* libtorch_tensor_grad(const LibtorchTensor* tensor);
    
    // Tensor information
    size_t libtorch_tensor_ndim(const LibtorchTensor* tensor);
    void libtorch_tensor_shape(const LibtorchTensor* tensor, int64_t* shape);
    size_t libtorch_tensor_numel(const LibtorchTensor* tensor);
    const float* libtorch_tensor_data_ptr(const LibtorchTensor* tensor);
    
    // Tensor comparison (for testing)
    bool libtorch_tensor_allclose(const LibtorchTensor* a, const LibtorchTensor* b, 
                                  double rtol, double atol);
    
    // Memory management
    void libtorch_tensor_free(LibtorchTensor* tensor);
    
    // Gradient operations
    void libtorch_tensor_set_grad(LibtorchTensor* tensor, const LibtorchTensor* grad);
    LibtorchTensor* libtorch_tensor_get_grad(const LibtorchTensor* tensor);
    void libtorch_tensor_zero_grad(LibtorchTensor* tensor);
    void libtorch_tensor_requires_grad(LibtorchTensor* tensor, bool requires_grad);
    
    // Adam optimizer functions
    typedef struct LibtorchAdam LibtorchAdam;
    
    // Create Adam optimizer
    LibtorchAdam* libtorch_adam_new(LibtorchTensor** parameters, size_t num_params, 
                                   double learning_rate, double beta1, double beta2, 
                                   double eps, double weight_decay, bool amsgrad);
    
    // Optimizer operations
    void libtorch_adam_step(LibtorchAdam* optimizer);
    void libtorch_adam_zero_grad(LibtorchAdam* optimizer);
    void libtorch_adam_set_learning_rate(LibtorchAdam* optimizer, double learning_rate);
    double libtorch_adam_get_learning_rate(const LibtorchAdam* optimizer);
    
    // Parameter access
    LibtorchTensor* libtorch_adam_get_parameter(const LibtorchAdam* optimizer, size_t index);
    size_t libtorch_adam_get_num_parameters(const LibtorchAdam* optimizer);
    
    // Cleanup
    void libtorch_adam_free(LibtorchAdam* optimizer);
    
    // Simple loss computation for validation
    LibtorchTensor* libtorch_compute_mse_loss(const LibtorchTensor* prediction, const LibtorchTensor* target);
    void libtorch_backward(LibtorchTensor* loss);
    
    // Error handling
    const char* libtorch_get_last_error();
    void libtorch_clear_error();
}