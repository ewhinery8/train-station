#include "cuda_wrapper.h"
#include <cstring>
#include <new>

// NOTE: These are CPU-side stubs to allow linking and feature-gated builds.
// Actual CUDA integration will replace these with real CUDA Runtime API calls and kernels.

static int g_current_device = -1;

extern "C" bool cuda_ffi_initialize() {
    // Stub: pretend initialization succeeds if at least one device (we report 0 here)
    return true;
}

extern "C" void cuda_ffi_cleanup() {
    // Stub: nothing to cleanup yet
}

extern "C" int get_cuda_device_count() {
    // Stub: report zero devices until real CUDA is wired
    return 0;
}

extern "C" bool set_cuda_device(int device_id) {
    // Stub: accept only device 0 when nonzero devices are present. For now, fail if device_id != 0.
    if (device_id == 0) {
        g_current_device = device_id;
        return true;
    }
    return false;
}

extern "C" int get_current_cuda_device() {
    return g_current_device;
}

extern "C" float* cuda_malloc(unsigned long long size) {
    // Stub: allocate on host to allow end-to-end tests to link
    if (size == 0) return nullptr;
    float* ptr = new (std::nothrow) float[size];
    return ptr;
}

extern "C" void cuda_free(float* ptr) {
    delete[] ptr;
}

extern "C" bool cuda_memcpy(float* dst, const float* src, unsigned long long size, int /*kind*/) {
    if (!dst || !src) return false;
    std::memcpy(dst, src, size * sizeof(float));
    return true;
}

extern "C" void* create_cuda_stream() {
    // Stub: just return a non-null placeholder pointer
    return reinterpret_cast<void*>(0x1);
}

extern "C" void destroy_cuda_stream(void* /*stream*/) {
    // Stub: nothing to destroy
}

extern "C" bool launch_add_kernel(const float* a, const float* b, float* result, unsigned long long size, void* /*stream*/) {
    if (!a || !b || !result) return false;
    for (unsigned long long i = 0; i < size; ++i) result[i] = a[i] + b[i];
    return true;
}

extern "C" bool launch_sub_kernel(const float* a, const float* b, float* result, unsigned long long size, void* /*stream*/) {
    if (!a || !b || !result) return false;
    for (unsigned long long i = 0; i < size; ++i) result[i] = a[i] - b[i];
    return true;
}

extern "C" bool launch_mul_kernel(const float* a, const float* b, float* result, unsigned long long size, void* /*stream*/) {
    if (!a || !b || !result) return false;
    for (unsigned long long i = 0; i < size; ++i) result[i] = a[i] * b[i];
    return true;
}

extern "C" bool launch_matmul_kernel(const float* a, const float* b, float* result, unsigned long long m, unsigned long long n, unsigned long long k, void* /*stream*/) {
    if (!a || !b || !result) return false;
    // Naive CPU matmul as placeholder
    for (unsigned long long i = 0; i < m; ++i) {
        for (unsigned long long j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (unsigned long long t = 0; t < k; ++t) {
                acc += a[i * k + t] * b[t * n + j];
            }
            result[i * n + j] = acc;
        }
    }
    return true;
}


