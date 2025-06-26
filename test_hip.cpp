#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_CALL(call) do {                    \
    hipError_t err = call;                     \
    if (err != hipSuccess) {                   \
        std::cerr << "HIP error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE);               \
    }                                          \
} while (0)

__global__ void add(int *x) {
    x[0] += 1;
}

int main() {
    int *d_x;
    int h_x = 41;

    HIP_CALL(hipMalloc(&d_x, sizeof(int)));
    HIP_CALL(hipMemcpy(d_x, &h_x, sizeof(int), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(add, dim3(1), dim3(1), 0, 0, d_x);
    HIP_CALL(hipMemcpy(&h_x, d_x, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CALL(hipFree(d_x));

    std::cout << "Result: " << h_x << std::endl;

    return 0;
}
