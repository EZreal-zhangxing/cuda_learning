#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<sys/time.h>
#include<cuda_runtime.h>
#include<cuComplex.h>
#include<cufft.h>
#include<cublas_v2.h>

#ifndef CUDA_KERNAL
#define CUDA_KERNAL

#define STREAM_MAX_SIZE 8

#define CHECK(call){                                                                                \
    if(call != cudaSuccess){                                                                        \
        printf("Line [%d],cudaError code [%d]: %s\n",__LINE__,call,cudaGetErrorString(call));       \
    }                                                                                               \
}                                                                                                   \

#define CHECK_STATUS(call){                                                                         \
    if(call != CUBLAS_STATUS_SUCCESS){                                                              \
        printf("Line [%d],cudaError code [%d]: %s\n",__LINE__,call,cublasGetStatusString(call));    \
    }                                                                                               \
}                                                                                                   \

void complex_fft_test(cuComplex * input,int paral,int input_size);

void float_invert(float * input,int input_size,int paral);

void complex_invert(cuComplex * input,int input_size,int paral);

void complex_matrix_gemm(cuComplex * matrix_a,cuComplex * matrix_b,cuComplex * matrix_c,int m,int n,int k);

void complex_matrix_transpose(cuComplex * matrix,int width,int height);

/**
 * ifft(fft(signal) * fft(coefficient))
*/
float match_filter(cuComplex * input_signal,cuComplex * coefficient,int input_size,int paral,int dimx,int dimy);

// data save as col first
float match_filter_col(cuComplex * input_signal,cuComplex * coefficient,int input_size,int paral,int dimx,int dimy);

float match_filter_rowfft_colkernal(cuComplex * input_signal,cuComplex * coefficient,int input_size,int paral,int dimx,int dimy);

float match_filter_streams(cuComplex * input_signal,cuComplex * coefficient,int input_size,int paral,int dimx,int dimy);

/**
 * conv(signal,coefficient)
*/
float conv_signal_coeff(cuComplex * input_signal,cuComplex * coefficient,cuComplex * output_signal,int signal_length,int batch,int co_input_size,int dimx,int dimy);

#endif