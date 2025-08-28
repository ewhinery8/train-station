#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Minimal CUDA FFI stubs for foundational linking. These do not use CUDA yet.

// Initialization / cleanup
bool cuda_ffi_initialize();
void cuda_ffi_cleanup();

// Device management
int get_cuda_device_count();
bool set_cuda_device(int device_id);
int get_current_cuda_device();

// Memory management
float* cuda_malloc(unsigned long long size);
void cuda_free(float* ptr);
bool cuda_memcpy(float* dst, const float* src, unsigned long long size, int kind);

// Streams
void* create_cuda_stream();
void destroy_cuda_stream(void* stream);

// Kernel launches (stubs)
bool launch_add_kernel(const float* a, const float* b, float* result, unsigned long long size, void* stream);
bool launch_sub_kernel(const float* a, const float* b, float* result, unsigned long long size, void* stream);
bool launch_mul_kernel(const float* a, const float* b, float* result, unsigned long long size, void* stream);
bool launch_matmul_kernel(const float* a, const float* b, float* result, unsigned long long m, unsigned long long n, unsigned long long k, void* stream);

#ifdef __cplusplus
}
#endif


