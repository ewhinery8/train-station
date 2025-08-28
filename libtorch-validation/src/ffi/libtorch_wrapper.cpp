#include "libtorch_wrapper.h"
#include <torch/torch.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

// Global error state for C interface
static thread_local std::string last_error;

// Helper function to set error and return nullptr
template<typename T>
T* set_error_and_return_null(const std::string& error) {
    last_error = error;
    return nullptr;
}

// Helper function to set error (for void functions)
void set_error(const std::string& error) {
    last_error = error;
}

// Helper function to catch exceptions and set error
#define LIBTORCH_TRY_CATCH(code) \
    try { \
        code \
    } catch (const std::exception& e) { \
        last_error = e.what(); \
        return nullptr; \
    } catch (...) { \
        last_error = "Unknown error occurred"; \
        return nullptr; \
    }

// Wrapper struct for torch::Tensor
struct LibtorchTensor {
    torch::Tensor tensor;
    
    LibtorchTensor(torch::Tensor t) : tensor(std::move(t)) {}
};

extern "C" {

// Tensor creation
LibtorchTensor* libtorch_tensor_new(const int64_t* shape, size_t ndim) {
    LIBTORCH_TRY_CATCH({
        std::vector<int64_t> sizes(shape, shape + ndim);
        auto tensor = torch::empty(sizes, torch::dtype(torch::kFloat32));
        return new LibtorchTensor(std::move(tensor));
    })
}

LibtorchTensor* libtorch_tensor_zeros(const int64_t* shape, size_t ndim) {
    LIBTORCH_TRY_CATCH({
        std::vector<int64_t> sizes(shape, shape + ndim);
        auto tensor = torch::zeros(sizes, torch::dtype(torch::kFloat32));
        return new LibtorchTensor(std::move(tensor));
    })
}

LibtorchTensor* libtorch_tensor_ones(const int64_t* shape, size_t ndim) {
    LIBTORCH_TRY_CATCH({
        std::vector<int64_t> sizes(shape, shape + ndim);
        auto tensor = torch::ones(sizes, torch::dtype(torch::kFloat32));
        return new LibtorchTensor(std::move(tensor));
    })
}

LibtorchTensor* libtorch_tensor_from_data(const float* data, const int64_t* shape, size_t ndim) {
    LIBTORCH_TRY_CATCH({
        std::vector<int64_t> sizes(shape, shape + ndim);
        
        // Calculate total number of elements
        int64_t numel = 1;
        for (size_t i = 0; i < ndim; ++i) {
            numel *= shape[i];
        }
        
        // Copy data to avoid lifetime issues
        auto tensor = torch::from_blob(
            const_cast<float*>(data), 
            sizes, 
            torch::dtype(torch::kFloat32)
        ).clone();
        
        return new LibtorchTensor(std::move(tensor));
    })
}

// Tensor operations
LibtorchTensor* libtorch_tensor_add_scalar(const LibtorchTensor* tensor, float scalar) {
    LIBTORCH_TRY_CATCH({
        auto result = tensor->tensor + scalar;
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_add_tensor(const LibtorchTensor* a, const LibtorchTensor* b) {
    LIBTORCH_TRY_CATCH({
        auto result = a->tensor + b->tensor;
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_sub_scalar(const LibtorchTensor* tensor, float scalar) {
    LIBTORCH_TRY_CATCH({
        auto result = tensor->tensor - scalar;
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_sub_tensor(const LibtorchTensor* a, const LibtorchTensor* b) {
    LIBTORCH_TRY_CATCH({
        auto result = a->tensor - b->tensor;
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_mul_scalar(const LibtorchTensor* tensor, float scalar) {
    LIBTORCH_TRY_CATCH({
        auto result = tensor->tensor * scalar;
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_mul_tensor(const LibtorchTensor* a, const LibtorchTensor* b) {
    LIBTORCH_TRY_CATCH({
        auto result = a->tensor * b->tensor;
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_div_scalar(const LibtorchTensor* tensor, float scalar) {
    LIBTORCH_TRY_CATCH({
        auto result = tensor->tensor / scalar;
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_div_tensor(const LibtorchTensor* a, const LibtorchTensor* b) {
    LIBTORCH_TRY_CATCH({
        auto result = a->tensor / b->tensor;
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_matmul(const LibtorchTensor* a, const LibtorchTensor* b) {
    LIBTORCH_TRY_CATCH({
        auto result = torch::matmul(a->tensor, b->tensor);
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_exp(const LibtorchTensor* a) {
    LIBTORCH_TRY_CATCH({
        auto result = torch::exp(a->tensor);
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_sqrt(const LibtorchTensor* a) {
    LIBTORCH_TRY_CATCH({
        auto result = torch::sqrt(a->tensor);
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_log(const LibtorchTensor* a) {
    LIBTORCH_TRY_CATCH({
        auto result = torch::log(a->tensor);
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_pow_scalar(const LibtorchTensor* a, float exponent) {
    LIBTORCH_TRY_CATCH({
        auto result = torch::pow(a->tensor, exponent);
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_pow_tensor(const LibtorchTensor* a, const LibtorchTensor* b) {
    LIBTORCH_TRY_CATCH({
        auto result = torch::pow(a->tensor, b->tensor);
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_softmax(const LibtorchTensor* a, int64_t dim) {
    LIBTORCH_TRY_CATCH({
        auto result = torch::softmax(a->tensor, dim);
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_relu(const LibtorchTensor* a) {
    LIBTORCH_TRY_CATCH({
        auto result = torch::relu(a->tensor);
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_tanh(const LibtorchTensor* a) {
    LIBTORCH_TRY_CATCH({
        auto result = torch::tanh(a->tensor);
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_sigmoid(const LibtorchTensor* a) {
    LIBTORCH_TRY_CATCH({
        auto result = torch::sigmoid(a->tensor);
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_leaky_relu(const LibtorchTensor* a, float negative_slope) {
    LIBTORCH_TRY_CATCH({
        auto result = torch::leaky_relu(a->tensor, negative_slope);
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_sum(const LibtorchTensor* a) {
    LIBTORCH_TRY_CATCH({
        auto result = a->tensor.sum();
        // Match our convention: represent scalars as shape [1]
        result = result.reshape({1}).contiguous();
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_mean(const LibtorchTensor* a) {
    LIBTORCH_TRY_CATCH({
        auto result = a->tensor.mean();
        result = result.reshape({1}).contiguous();
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_norm(const LibtorchTensor* a) {
    LIBTORCH_TRY_CATCH({
        auto result = a->tensor.norm();
        result = result.reshape({1}).contiguous();
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_std(const LibtorchTensor* a, bool unbiased) {
    LIBTORCH_TRY_CATCH({
        auto result = a->tensor.std(unbiased);
        result = result.reshape({1}).contiguous();
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_var(const LibtorchTensor* a, bool unbiased) {
    LIBTORCH_TRY_CATCH({
        auto result = a->tensor.var(unbiased);
        result = result.reshape({1}).contiguous();
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_sum_dims(const LibtorchTensor* a, const int64_t* dims, size_t ndims, bool keepdim) {
    LIBTORCH_TRY_CATCH({
        std::vector<int64_t> vd(dims, dims + ndims);
        auto result = a->tensor.sum(vd, keepdim);
        if (result.dim() == 0) { result = result.reshape({1}); }
        result = result.contiguous();
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_mean_dims(const LibtorchTensor* a, const int64_t* dims, size_t ndims, bool keepdim) {
    LIBTORCH_TRY_CATCH({
        std::vector<int64_t> vd(dims, dims + ndims);
        auto result = a->tensor.mean(vd, keepdim);
        if (result.dim() == 0) { result = result.reshape({1}); }
        result = result.contiguous();
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_norm_dims(const LibtorchTensor* a, const int64_t* dims, size_t ndims, bool keepdim) {
    LIBTORCH_TRY_CATCH({
        std::vector<int64_t> vd(dims, dims + ndims);
        auto result = torch::norm(a->tensor, 2, vd, keepdim);
        if (result.dim() == 0) { result = result.reshape({1}); }
        result = result.contiguous();
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_std_dims(const LibtorchTensor* a, const int64_t* dims, size_t ndims, bool keepdim, bool unbiased) {
    LIBTORCH_TRY_CATCH({
        std::vector<int64_t> vd(dims, dims + ndims);
        auto result = torch::std(a->tensor, vd, unbiased, keepdim);
        if (result.dim() == 0) { result = result.reshape({1}); }
        result = result.contiguous();
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_var_dims(const LibtorchTensor* a, const int64_t* dims, size_t ndims, bool keepdim, bool unbiased) {
    LIBTORCH_TRY_CATCH({
        std::vector<int64_t> vd(dims, dims + ndims);
        auto result = torch::var(a->tensor, vd, unbiased, keepdim);
        if (result.dim() == 0) { result = result.reshape({1}); }
        result = result.contiguous();
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_min(const LibtorchTensor* a) {
    LIBTORCH_TRY_CATCH({
        auto result = a->tensor.min();
        if (result.dim() == 0) { result = result.reshape({1}); }
        result = result.contiguous();
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_min_dims(const LibtorchTensor* a, const int64_t* dims, size_t ndims, bool keepdim) {
    LIBTORCH_TRY_CATCH({
        std::vector<int64_t> vd(dims, dims + ndims);
        auto result = torch::amin(a->tensor, vd, keepdim);
        if (result.dim() == 0) { result = result.reshape({1}); }
        result = result.contiguous();
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_max(const LibtorchTensor* a) {
    LIBTORCH_TRY_CATCH({
        auto result = a->tensor.max();
        if (result.dim() == 0) { result = result.reshape({1}); }
        result = result.contiguous();
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_max_dims(const LibtorchTensor* a, const int64_t* dims, size_t ndims, bool keepdim) {
    LIBTORCH_TRY_CATCH({
        std::vector<int64_t> vd(dims, dims + ndims);
        auto result = torch::amax(a->tensor, vd, keepdim);
        if (result.dim() == 0) { result = result.reshape({1}); }
        result = result.contiguous();
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_argmin(const LibtorchTensor* a) {
    LIBTORCH_TRY_CATCH({
        auto result = std::get<1>(a->tensor.min( /*dim*/ 0, /*keepdim*/ false));
        // For global argmin, flatten and use argmin to get index
        auto flat = a->tensor.reshape({-1});
        auto idx = std::get<1>(flat.min(0));
        auto idx_f = idx.to(torch::kFloat32).reshape({1}).contiguous();
        return new LibtorchTensor(std::move(idx_f));
    })
}

LibtorchTensor* libtorch_tensor_argmin_dim(const LibtorchTensor* a, int64_t dim, bool keepdim) {
    LIBTORCH_TRY_CATCH({
        auto tup = torch::min(a->tensor, dim, /*keepdim*/ keepdim);
        auto idx = std::get<1>(tup).to(torch::kFloat32).contiguous();
        if (idx.dim() == 0) { idx = idx.reshape({1}); }
        return new LibtorchTensor(std::move(idx));
    })
}

LibtorchTensor* libtorch_tensor_argmax(const LibtorchTensor* a) {
    LIBTORCH_TRY_CATCH({
        auto flat = a->tensor.reshape({-1});
        auto tup = torch::max(flat, 0);
        auto idx = std::get<1>(tup).to(torch::kFloat32).reshape({1}).contiguous();
        return new LibtorchTensor(std::move(idx));
    })
}

LibtorchTensor* libtorch_tensor_argmax_dim(const LibtorchTensor* a, int64_t dim, bool keepdim) {
    LIBTORCH_TRY_CATCH({
        auto tup = torch::max(a->tensor, dim, keepdim);
        auto idx = std::get<1>(tup).to(torch::kFloat32).contiguous();
        if (idx.dim() == 0) { idx = idx.reshape({1}); }
        return new LibtorchTensor(std::move(idx));
    })
}

LibtorchTensor* libtorch_tensor_cat(const LibtorchTensor** tensors, size_t count, int64_t dim) {
    LIBTORCH_TRY_CATCH({
        std::vector<torch::Tensor> vec;
        vec.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            vec.push_back(tensors[i]->tensor);
        }
        auto result = torch::cat(vec, dim);
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_stack(const LibtorchTensor** tensors, size_t count, int64_t dim) {
    LIBTORCH_TRY_CATCH({
        std::vector<torch::Tensor> vec;
        vec.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            vec.push_back(tensors[i]->tensor);
        }
        auto result = torch::stack(vec, dim);
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_view(const LibtorchTensor* tensor, const int64_t* shape, size_t ndim) {
    LIBTORCH_TRY_CATCH({
        std::vector<int64_t> sizes(shape, shape + ndim);
        // Ensure contiguity like PyTorch's view requirement
        auto base = tensor->tensor;
        if (!base.is_contiguous()) {
            base = base.contiguous();
        }
        auto result = base.view(sizes);
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_permute(const LibtorchTensor* tensor, const int64_t* dims, size_t ndim) {
    LIBTORCH_TRY_CATCH({
        std::vector<int64_t> order(dims, dims + ndim);
        auto result = tensor->tensor.permute(order).contiguous();
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_index_select(const LibtorchTensor* tensor, int64_t dim, const int64_t* indices, size_t nindices) {
    LIBTORCH_TRY_CATCH({
        std::vector<int64_t> idx(indices, indices + nindices);
        auto idx_tensor = torch::from_blob(idx.data(), {(long long)nindices}, torch::dtype(torch::kLong)).clone();
        auto result = torch::index_select(tensor->tensor, dim, idx_tensor);
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_gather(const LibtorchTensor* tensor, int64_t dim, const int64_t* index_data, const int64_t* index_shape, size_t index_ndim) {
    LIBTORCH_TRY_CATCH({
        std::vector<int64_t> ishape(index_shape, index_shape + index_ndim);
        int64_t numel = 1;
        for (size_t i = 0; i < index_ndim; ++i) numel *= ishape[i];
        // Build index tensor from raw data and shape, dtype Long
        auto idx = torch::from_blob(const_cast<int64_t*>(index_data), ishape, torch::dtype(torch::kLong)).clone();
        auto result = torch::gather(tensor->tensor, dim, idx);
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_masked_fill(const LibtorchTensor* tensor, const bool* mask, size_t numel, float value) {
    LIBTORCH_TRY_CATCH({
        // Build a boolean mask tensor of same shape as input
        auto sizes = tensor->tensor.sizes();
        auto mask_tensor = torch::empty(sizes, torch::dtype(torch::kBool));
        auto mask_ptr = mask_tensor.data_ptr<bool>();
        for (size_t i = 0; i < numel; ++i) {
            mask_ptr[i] = mask[i];
        }
        auto result = tensor->tensor.masked_fill(mask_tensor, value);
        return new LibtorchTensor(std::move(result));
    })
}

LibtorchTensor* libtorch_tensor_select(const LibtorchTensor* tensor, int64_t dim, int64_t index) {
    LIBTORCH_TRY_CATCH({
        auto result = torch::select(tensor->tensor, dim, index).contiguous();
        return new LibtorchTensor(std::move(result));
    })
}

// Gradient operations
LibtorchTensor* libtorch_tensor_require_grad(LibtorchTensor* tensor, bool requires_grad) {
    LIBTORCH_TRY_CATCH({
        tensor->tensor.requires_grad_(requires_grad);
        return tensor; // Return the same tensor for chaining
    })
}

void libtorch_tensor_backward(LibtorchTensor* tensor, LibtorchTensor* grad_output) {
    try {
        if (grad_output != nullptr) {
            tensor->tensor.backward(grad_output->tensor);
        } else {
            tensor->tensor.backward();
        }
    } catch (const std::exception& e) {
        last_error = e.what();
    } catch (...) {
        last_error = "Unknown error occurred during backward";
    }
}

LibtorchTensor* libtorch_tensor_grad(const LibtorchTensor* tensor) {
    LIBTORCH_TRY_CATCH({
        if (!tensor->tensor.grad().defined()) {
            return nullptr;
        }
        auto grad = tensor->tensor.grad();
        return new LibtorchTensor(std::move(grad));
    })
}

// Tensor information
size_t libtorch_tensor_ndim(const LibtorchTensor* tensor) {
    try {
        return tensor->tensor.dim();
    } catch (...) {
        return 0;
    }
}

void libtorch_tensor_shape(const LibtorchTensor* tensor, int64_t* shape) {
    try {
        auto sizes = tensor->tensor.sizes();
        for (size_t i = 0; i < sizes.size(); ++i) {
            shape[i] = sizes[i];
        }
    } catch (const std::exception& e) {
        last_error = e.what();
    }
}

size_t libtorch_tensor_numel(const LibtorchTensor* tensor) {
    try {
        return tensor->tensor.numel();
    } catch (...) {
        return 0;
    }
}

const float* libtorch_tensor_data_ptr(const LibtorchTensor* tensor) {
    try {
        return tensor->tensor.data_ptr<float>();
    } catch (...) {
        last_error = "Failed to get data pointer";
        return nullptr;
    }
}

// Tensor comparison
bool libtorch_tensor_allclose(const LibtorchTensor* a, const LibtorchTensor* b, 
                              double rtol, double atol) {
    try {
        return torch::allclose(a->tensor, b->tensor, rtol, atol);
    } catch (...) {
        last_error = "Failed to compare tensors";
        return false;
    }
}

// Memory management
void libtorch_tensor_free(LibtorchTensor* tensor) {
    delete tensor;
}

// Error handling
const char* libtorch_get_last_error() {
    return last_error.c_str();
}

void libtorch_clear_error() {
    last_error.clear();
}

// Gradient operations
void libtorch_tensor_set_grad(LibtorchTensor* tensor, const LibtorchTensor* grad) {
    if (!tensor || !grad) {
        set_error("Null tensor pointer in set_grad");
        return;
    }
    
    try {
        tensor->tensor.set_requires_grad(true);
        if (tensor->tensor.grad().defined()) {
            tensor->tensor.mutable_grad() = grad->tensor.clone();
        } else {
            tensor->tensor.mutable_grad() = grad->tensor.clone();
        }
    } catch (const std::exception& e) {
        set_error(std::string("Error setting gradient: ") + e.what());
    }
}

LibtorchTensor* libtorch_tensor_get_grad(const LibtorchTensor* tensor) {
    if (!tensor) {
        set_error("Null tensor pointer in get_grad");
        return nullptr;
    }
    
    try {
        if (!tensor->tensor.grad().defined()) {
            return nullptr;
        }
        
        return new LibtorchTensor{tensor->tensor.grad().clone()};
    } catch (const std::exception& e) {
        set_error(std::string("Error getting gradient: ") + e.what());
        return nullptr;
    }
}

void libtorch_tensor_zero_grad(LibtorchTensor* tensor) {
    if (!tensor) {
        set_error("Null tensor pointer in zero_grad");
        return;
    }
    
    try {
        if (tensor->tensor.grad().defined()) {
            tensor->tensor.mutable_grad().zero_();
        }
    } catch (const std::exception& e) {
        set_error(std::string("Error zeroing gradient: ") + e.what());
    }
}

void libtorch_tensor_requires_grad(LibtorchTensor* tensor, bool requires_grad) {
    if (!tensor) {
        set_error("Null tensor pointer in requires_grad");
        return;
    }
    
    try {
        tensor->tensor.set_requires_grad(requires_grad);
    } catch (const std::exception& e) {
        set_error(std::string("Error setting requires_grad: ") + e.what());
    }
}

// Adam optimizer implementation
struct LibtorchAdam {
    torch::optim::Adam optimizer;
    
    LibtorchAdam(std::vector<torch::Tensor> params, const torch::optim::AdamOptions& options)
        : optimizer(params, options) {}
};

LibtorchAdam* libtorch_adam_new(LibtorchTensor** parameters, size_t num_params, 
                               double learning_rate, double beta1, double beta2, 
                               double eps, double weight_decay, bool amsgrad) {
    try {
        std::vector<torch::Tensor> param_tensors;
        param_tensors.reserve(num_params);
        
        for (size_t i = 0; i < num_params; ++i) {
            if (!parameters[i] || !parameters[i]->tensor.defined()) {
                set_error("Invalid parameter tensor at index " + std::to_string(i));
                return nullptr;
            }
            
            // Ensure the tensor requires gradients
            parameters[i]->tensor.set_requires_grad(true);
            
            // Store the exact same tensor reference (no copying)
            param_tensors.push_back(parameters[i]->tensor);
        }
        
        // Create Adam options
        torch::optim::AdamOptions options(learning_rate);
        options.betas(std::make_tuple(beta1, beta2));
        options.eps(eps);
        options.weight_decay(weight_decay);
        options.amsgrad(amsgrad);
        
        return new LibtorchAdam(param_tensors, options);
        
    } catch (const std::exception& e) {
        set_error(std::string("Error creating Adam optimizer: ") + e.what());
        return nullptr;
    }
}

void libtorch_adam_step(LibtorchAdam* optimizer) {
    if (!optimizer) {
        set_error("Null optimizer pointer");
        return;
    }
    
    try {
        optimizer->optimizer.step();
    } catch (const std::exception& e) {
        set_error(std::string("Error in Adam step: ") + e.what());
    }
}

void libtorch_adam_zero_grad(LibtorchAdam* optimizer) {
    if (!optimizer) {
        set_error("Null optimizer pointer");
        return;
    }
    
    try {
        optimizer->optimizer.zero_grad();
    } catch (const std::exception& e) {
        set_error(std::string("Error in Adam zero_grad: ") + e.what());
    }
}

void libtorch_adam_set_learning_rate(LibtorchAdam* optimizer, double learning_rate) {
    if (!optimizer) {
        set_error("Null optimizer pointer");
        return;
    }
    
    try {
        for (auto& param_group : optimizer->optimizer.param_groups()) {
            param_group.options().set_lr(learning_rate);
        }
    } catch (const std::exception& e) {
        set_error(std::string("Error setting learning rate: ") + e.what());
    }
}

double libtorch_adam_get_learning_rate(const LibtorchAdam* optimizer) {
    if (!optimizer) {
        set_error("Null optimizer pointer");
        return 0.0;
    }
    
    try {
        if (!optimizer->optimizer.param_groups().empty()) {
            return optimizer->optimizer.param_groups()[0].options().get_lr();
        }
        return 0.0;
    } catch (const std::exception& e) {
        set_error(std::string("Error getting learning rate: ") + e.what());
        return 0.0;
    }
}

LibtorchTensor* libtorch_adam_get_parameter(const LibtorchAdam* optimizer, size_t index) {
    if (!optimizer) {
        set_error("Null optimizer pointer");
        return nullptr;
    }
    
    try {
        if (optimizer->optimizer.param_groups().empty()) {
            set_error("No parameter groups in optimizer");
            return nullptr;
        }
        
        auto& param_group = optimizer->optimizer.param_groups()[0]; // Get first parameter group
        if (index >= param_group.params().size()) {
            set_error("Parameter index out of bounds");
            return nullptr;
        }
        
        return new LibtorchTensor{param_group.params()[index]}; // Return the parameter at the given index
    } catch (const std::exception& e) {
        set_error(std::string("Error getting parameter: ") + e.what());
        return nullptr;
    }
}

size_t libtorch_adam_get_num_parameters(const LibtorchAdam* optimizer) {
    if (!optimizer) {
        set_error("Null optimizer pointer");
        return 0;
    }
    
    if (optimizer->optimizer.param_groups().empty()) {
        return 0;
    }
    
    return optimizer->optimizer.param_groups()[0].params().size();
}

void libtorch_adam_free(LibtorchAdam* optimizer) {
    delete optimizer;
}

// Simple loss computation for autograd validation
LibtorchTensor* libtorch_compute_mse_loss(const LibtorchTensor* prediction, const LibtorchTensor* target) {
    if (!prediction || !target) {
        set_error("Null tensor pointer in MSE loss computation");
        return nullptr;
    }
    
    try {
        // Compute MSE loss: mean((prediction - target)^2)
        torch::Tensor loss = torch::mse_loss(prediction->tensor, target->tensor);
        return new LibtorchTensor{loss};
    } catch (const std::exception& e) {
        set_error(std::string("Error computing MSE loss: ") + e.what());
        return nullptr;
    }
}

void libtorch_backward(LibtorchTensor* loss) {
    if (!loss) {
        set_error("Null loss tensor pointer");
        return;
    }
    
    try {
        loss->tensor.backward();
    } catch (const std::exception& e) {
        set_error(std::string("Error in backward pass: ") + e.what());
    }
}

} // extern "C"