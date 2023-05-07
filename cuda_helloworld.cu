#include<stdio.h>
#include<iostream>
using namespace std;
#include<cuda_runtime.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if(error != cudaSuccess){\
        printf("Error: %s:%d ,",__FILE__,__LINE__);\
        printf("Code: %d, reason: %s \n",error,cudaGetErrorString(error));\
        exit(-10*error);\
    }\
}

__global__ void helloFromGpu(void){
    printf("hello world from gpus! and block Id_x is  %d thread Id_x is %d \n",blockIdx.x,threadIdx.x);
    // cout << "hello world from gpus!" << endl;
}

int main(void){
    int dev =0;
    cout << "hello world from cpu!" << endl;
    cudaDeviceProp cudeProp;
    CHECK(cudaGetDeviceProperties(&cudeProp,dev));
    printf("Using Device %d: %s\n",dev,cudeProp.name);
    CHECK(cudaSetDevice(dev));
    helloFromGpu <<<2,10>>>();
    cudaError_t error;
    error = cudaGetLastError();
    cout << "cuda status is " << cudaGetErrorString(error) <<  "["<< (error == cudaSuccess) <<"]" << endl;
    error = cudaDeviceReset();
    cout << "cuda status is " << cudaGetErrorString(error) <<  "["<< (error == cudaSuccess) <<"]" << endl;
    return 0;
}