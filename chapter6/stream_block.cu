#include<cuda_runtime.h>
#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/time.h>
#include<device_functions.h>

long seconds(){
    timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec * 1e6 + t.tv_usec;
}

__global__ void kernal(int sign){
    printf("kernal execute stream id %d\n",sign);
}

__global__ void kernal_1(int sign){
    printf("kernal execute stream id %d\n",sign);
}
__global__ void kernal_2(int sign){
    printf("kernal execute stream id %d\n",sign);
}
__global__ void kernal_3(int sign){
    printf("kernal execute stream id %d\n",sign);
}
__global__ void kernal_4(int sign){
    printf("kernal execute stream id %d\n",sign);
}

void blocking_stream(){
    cudaStream_t streams[2];
    cudaStreamCreate(streams);
    cudaStreamCreate(streams + 1);

    kernal<<<1,1,0,streams[0]>>>(0);
    kernal<<<1,1>>>(-1);
    kernal<<<1,1,0,streams[1]>>>(1);

    printf("host print \n");
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
}

void non_blocking_stream(){
    cudaStream_t streams[2];
    cudaStreamCreateWithFlags(streams,cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(streams + 1,cudaStreamNonBlocking);

    kernal<<<1,1,0,streams[0]>>>(0);
    kernal<<<1,1>>>(-1);
    kernal<<<1,1,0,streams[1]>>>(1);

    printf("host print \n");
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
}

void event_syn(){
    cudaStream_t streams[4];
    cudaStreamCreate(streams);
    cudaStreamCreate(streams + 1);
    cudaStreamCreate(streams + 2);
    cudaStreamCreate(streams + 3);

    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for(int i=0;i<4;i++){
        kernal<<<1,1,0,streams[i]>>>(i);
        kernal<<<1,1,0,streams[i]>>>(i);
        // kernal<<<1,1>>>(i);
        kernal<<<1,1,0,streams[i]>>>(i);
        kernal<<<1,1,0,streams[i]>>>(i);
    }
    
    cudaEventRecord(end);

    cudaEventSynchronize(end);
    float times = 0.0;
    cudaEventElapsedTime(&times,start,end);
    printf("all elapsed time %f \n",times);
    for(int i=0;i<4;i++){
        cudaStreamDestroy(streams[i]);
    }

}

void deep_first(){
    cudaStream_t streams[4];
    cudaStreamCreate(streams);
    cudaStreamCreate(streams + 1);
    cudaStreamCreate(streams + 2);
    cudaStreamCreate(streams + 3);

    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for(int i=0;i<4;i++){
        kernal_1<<<1,1,0,streams[i]>>>(i);
        kernal_2<<<1,1,0,streams[i]>>>(i);
        kernal_3<<<1,1,0,streams[i]>>>(i);
        kernal_4<<<1,1,0,streams[i]>>>(i);
    }
    
    cudaEventRecord(end);

    cudaEventSynchronize(end);
    float times = 0.0;
    cudaEventElapsedTime(&times,start,end);
    printf("all elapsed time %f \n",times);
    for(int i=0;i<4;i++){
        cudaStreamDestroy(streams[i]);
    }

}

void breadth_first(){
    cudaStream_t streams[4];
    cudaStreamCreate(streams);
    cudaStreamCreate(streams + 1);
    cudaStreamCreate(streams + 2);
    cudaStreamCreate(streams + 3);

    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    for(int i=0;i<4;i++){
        kernal_1<<<1,1,0,streams[i]>>>(i);
    }
    for(int i=0;i<4;i++){
        kernal_2<<<1,1,0,streams[i]>>>(i);
    }
    for(int i=0;i<4;i++){
        kernal_3<<<1,1,0,streams[i]>>>(i);
    }
    for(int i=0;i<4;i++){
        kernal_4<<<1,1,0,streams[i]>>>(i);
    }
    cudaEventRecord(end);

    cudaEventSynchronize(end);
    float times = 0.0;
    cudaEventElapsedTime(&times,start,end);
    printf("all elapsed time %f \n",times);
    for(int i=0;i<4;i++){
        cudaStreamDestroy(streams[i]);
    }

}

void CUDART_CB callback(cudaStream_t stream,cudaError_t status,void * data){
    printf("call back execute \n");
}

void openmp(){
    cudaStream_t streams[4];
    cudaStreamCreate(streams);
    cudaStreamCreate(streams + 1);
    cudaStreamCreate(streams + 2);
    cudaStreamCreate(streams + 3);

    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    omp_set_num_threads(4);
    #pragma omp parallel
    {   
        int i = omp_get_thread_num();
        kernal_1<<<1,1,0,streams[i]>>>(i);
        kernal_2<<<1,1,0,streams[i]>>>(i);
        kernal_3<<<1,1,0,streams[i]>>>(i);
        kernal_4<<<1,1,0,streams[i]>>>(i);
        cudaStreamAddCallback(streams[i],callback,0,0);
    }

    cudaEventRecord(end);

    cudaEventSynchronize(end);
    float times = 0.0;
    cudaEventElapsedTime(&times,start,end);
    printf("all elapsed time %f \n",times);
    for(int i=0;i<4;i++){
        cudaStreamDestroy(streams[i]);
    }

}

int main(int argv,char * argc[]){

    // blocking_stream();
    // non_blocking_stream();
    // event_syn();
    // deep_first();
    // cudaDeviceSynchronize();
    // breadth_first();
    openmp();
    cudaDeviceReset();
    return 0;
}