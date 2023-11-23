#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/time.h>
#include<cufft.h>
#include<unistd.h>
void random_init(cuComplex * matrix ,int size){
    time_t t;
    srand((unsigned int)time(&t));
    for(int i=0;i<size;i++){
        // matrix[i] = (rand() & 0xFFFFF) / 100;
        matrix[i].x = i;
        matrix[i].y = i;
    }
}


int main(int argc,char * argv[]){
    int nstream = 4;
    cuComplex * h_data[nstream],* d_data,* d_output,* h_output;
    int nums = 10,batch = 128;
    if(argc > 1){
        nums = atoi(argv[1]);
    }
    cudaMallocHost((void **)&h_data[0],sizeof(cuComplex) * nums * batch/nstream);
    cudaMallocHost((void **)&h_data[1],sizeof(cuComplex) * nums * batch/nstream);
    cudaMallocHost((void **)&h_data[2],sizeof(cuComplex) * nums * batch/nstream);
    cudaMallocHost((void **)&h_data[3],sizeof(cuComplex) * nums * batch/nstream);
    cudaMalloc((void **)&d_output,sizeof(cuComplex) * nums * batch);
    cudaMalloc((void **)&d_data,sizeof(cuComplex) * nums * batch);
    h_output = (cuComplex *)malloc(sizeof(cuComplex) * nums * batch);
    random_init(h_data[0],nums * batch/nstream);
    random_init(h_data[1],nums * batch/nstream);
    random_init(h_data[2],nums * batch/nstream);
    random_init(h_data[3],nums * batch/nstream);
    
    
    cufftHandle handle[nstream];
    cudaStream_t streams[nstream];
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    for(int i=0;i<nstream;i++){
        cufftCreate(&handle[i]);
        cudaStreamCreate(&streams[i]);
    }

    int n[1] = {nums};
    int inembed[2] = {nums,batch/nstream};
    int onembed[2] = {nums,batch/nstream};
    for(int i=0;i<nstream;i++){
        cufftPlanMany(&handle[i],1,n,inembed,1,nums,onembed,1,nums,CUFFT_C2C,batch/nstream);
        cufftSetStream(handle[i],streams[i]);
    }
    cudaEventRecord(start);
    for(int i=0;i<nstream;i++){
        cudaMemcpyAsync(d_data + i * (batch/nstream) * nums,h_data[i],(batch/nstream) * nums * sizeof(cuComplex),cudaMemcpyHostToDevice,streams[i]);
        cufftExecC2C(handle[i],d_data + i * (batch/nstream) * nums,d_output + i * (batch/nstream) * nums,CUFFT_FORWARD);
        // cufftExecC2C(handle[i],h_data + i * (batch/nstream) * nums,d_output + i * (batch/nstream) * nums,CUFFT_INVERSE);
        // usleep(2000);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float times = 0;
    cudaEventElapsedTime(&times,start,end);
    printf("complex match filter save with rows elapsed time %f us \n",times * 1000.0);
    
    // cudaMemcpy(h_output,d_output,sizeof(cuComplex) * nums * batch,cudaMemcpyDeviceToHost);
    // for(int i=0;i<batch;i++){
    //     for(int j=0;j<nums;j++){
    //         printf("fft(%5.2f,%5.2f i) = (%5.2f,%5.2f i) ",h_data[i * nums + j].x,h_data[i * nums + j].y,h_output[i * nums + j].x,h_output[i * nums + j].y);
    //     }  
    //     printf("\n");
    // }
    cudaFreeHost(h_data);
    cudaFree(d_output);
    free(h_output);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return 0;
}