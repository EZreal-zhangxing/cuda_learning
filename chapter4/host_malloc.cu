#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<sys/time.h>


long seconds(){
    timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec * 1e6 + t.tv_usec;
}

void normal_malloc(int size,int bytes){
    int * h_array = (int *)malloc(sizeof(bytes));

    h_array[0] = 1024;
    int * d_array;
    cudaMalloc((void **)&d_array,bytes);
    cudaMemcpy(d_array,h_array,bytes,cudaMemcpyHostToDevice);

    free(h_array);
    cudaFree(d_array);
    cudaDeviceReset();
}

void host_malloc(int size,int bytes){
    int * h_array ,* d_array;
    cudaMallocHost((void **)&h_array,bytes);
    h_array[0] = 1024;

    cudaMalloc((void **)&d_array,bytes);
    cudaMemcpy(d_array,h_array,bytes,cudaMemcpyHostToDevice);

    cudaFreeHost(h_array);
    cudaFree(d_array);
    cudaDeviceReset();
}

int main(int argc,char * argv[]){
    int size = 1<<14,bytes = 0;
    if(argc > 1){
        size = atoi(argv[1]);
    }
    bytes = sizeof(int) * size;
    long start = seconds();
    normal_malloc(size,bytes);
    printf("normal malloc using %f ms \n",(seconds() - start)/1000.0);

    start = seconds();
    host_malloc(size,bytes);
    printf("host malloc using %f ms \n",(seconds() - start)/1000.0);
    return 0;
}