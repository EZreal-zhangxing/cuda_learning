#include<stdio.h>
#include<cuda_runtime.h>


__global__ void helloFromGpu(){
    printf("Hello world from GPU!\n");
}

int main(){
    printf("Hello world from CPU!\n");
    helloFromGpu<<<1,10>>>();

    printf("CPU print out finished!\n");
    return 0;
}