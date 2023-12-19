#ifndef COMMON_MACROS_HEADER
#define COMMON_MACROS_HEADER


#if defined ( FFTX_CUDA )

#include <cufft.h>
#include <cufftXt.h>
#include <helper_cuda.h>

#define DEVICE_FFT_DOUBLEREAL cufftDoubleReal
#define DEVICE_MALLOC cudaMalloc
#define DEVICE_FREE cudaFree
#define DEVICE_MEM_COPY cudaMemcpy
#define DEVICE_MEM_SET cudaMemset
#define MEM_COPY_DEVICE_TO_DEVICE cudaMemcpyDeviceToDevice
#define MEM_COPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
#define MEM_COPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
#define DEVICE_ERROR_T cudaError_t
#define DEVICE_GET_LAST_ERROR cudaGetLastError
#define DEVICE_GET_ERROR_STRING cudaGetErrorString
#define DEVICE_SUCCESS cudaSuccess
#define DEVICE_SYNCHRONIZE cudaDeviceSynchronize
#define DEVICE_SET_CACHE_CONFIG cudaDeviceSetCacheConfig
#define DEVICE_CACHE_PREFER_SHARED cudaFuncCachePreferShared
#define DEVICE_EVENT_T cudaEvent_t
#define DEVICE_EVENT_CREATE cudaEventCreate
#define DEVICE_EVENT_RECORD cudaEventRecord
#define DEVICE_EVENT_ELAPSED_TIME cudaEventElapsedTime
#define DEVICE_EVENT_SYNCHRONIZE cudaEventSynchronize

#elif defined ( FFTX_HIP )

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#include <hipfft.h>
#include <rocfft/rocfft.h>

#define DEVICE_FFT_DOUBLEREAL hipfftDoubleReal
#define DEVICE_MALLOC hipMalloc
#define DEVICE_FREE hipFree
#define DEVICE_MEM_COPY hipMemcpy
#define DEVICE_MEM_SET hipMemset
#define MEM_COPY_DEVICE_TO_DEVICE hipMemcpyDeviceToDevice
#define MEM_COPY_DEVICE_TO_HOST hipMemcpyDeviceToHost
#define MEM_COPY_HOST_TO_DEVICE hipMemcpyHostToDevice
#define DEVICE_ERROR_T hipError_t
#define DEVICE_GET_LAST_ERROR hipGetLastError
#define DEVICE_GET_ERROR_STRING hipGetErrorString
#define DEVICE_SUCCESS hipSuccess
#define DEVICE_SYNCHRONIZE hipDeviceSynchronize
#define DEVICE_SET_CACHE_CONFIG hipDeviceSetCacheConfig
#define DEVICE_CACHE_PREFER_SHARED hipFuncCachePreferShared
#define DEVICE_EVENT_T hipEvent_t
#define DEVICE_EVENT_CREATE hipEventCreate
#define DEVICE_EVENT_RECORD hipEventRecord
#define DEVICE_EVENT_ELAPSED_TIME hipEventElapsedTime
#define DEVICE_EVENT_SYNCHRONIZE hipEventSynchronize

#endif

// Functions that are defined if and only if either CUDA or HIP.
#if defined ( FFTX_CUDA ) || defined ( FFTX_HIP )

#include <iostream>

inline void DEVICE_CHECK_ERROR ( DEVICE_ERROR_T a_rc )
{
    // There does not appear to be a HIP analogue.
  #if defined(__CUDACC__)
    checkCudaErrors(a_rc);
  #endif
    if (a_rc != DEVICE_SUCCESS) {
        std::cerr << "Failure with code " << a_rc
                  << " meaning " << DEVICE_GET_ERROR_STRING(a_rc)
                  << std::endl;
        exit(-1);
    }
}

#endif                    //  defined ( FFTX_CUDA ) || defined ( FFTX_HIP )

#endif                    // COMMON_MACROS_HEADER
