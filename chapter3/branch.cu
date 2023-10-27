#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/time.h>
// smsp__sass_average_branch_targets_threads_uniform.pct

long seconds(){
    timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec*1e6 + t.tv_usec;
}

__global__ void warmingUp(float * sum){
    int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    if((thread_idx) % 2 == 0){
        sum[thread_idx] += 100;
    }else{
        sum[thread_idx] += 200;
    }
}

__global__ void mathKernal2(float * sum){
    int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    if(thread_idx%2 == 0){
        sum[thread_idx] += 100;
    }else{
        sum[thread_idx] += 200;
    }
}

// 线程束粒度的分化
__global__ void mathKernal3(float * sum){
    int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    if((thread_idx / warpSize )%2 == 0){
        sum[thread_idx] += 100;
    }else{
        sum[thread_idx] += 200;
    }
}

__global__ void mathKernal4(float * sum){
    int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    bool ispred = thread_idx%2 == 0;
    if(ispred){
        sum[thread_idx] += 100;
    }
    if(!ispred){
        sum[thread_idx] += 200;
    }
}
int main(){
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,device);
    printf("Device Properties: \n");
    printf(" -- Device name: %s \n",prop.name);
    printf(" -- Device major: %d.%d \n",prop.major,prop.minor);
    printf(" -- Device warpSize: %d \n",prop.warpSize);
    printf(" -- Device maxThreadsPerMultiProcessor: %d \n",prop.maxThreadsPerMultiProcessor);
    printf(" -- Device maxThreadsPerBlock: %d \n",prop.maxThreadsPerBlock);
    printf(" -- Device maxThreadsDim %dx%dx%d \n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
    printf(" -- Device maxGridSize %dx%dx%d \n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
    printf(" -- Device memoryBusWidth %d bits \n",prop.memoryBusWidth);
    printf(" -- Device totalGlobalMem %ld MB \n",prop.totalGlobalMem / (1024*1024));
    const int nums = 1 << 24,bytes = nums * sizeof(float);
    float * h_matrix = (float *)malloc(bytes);
    memset(h_matrix,0,bytes);

    float * d_matrix;
    cudaMalloc((void **)&d_matrix,bytes);


    dim3 block(128);
    dim3 grid((nums + block.x - 1)/block.x);

    long start = seconds();
    warmingUp<<<grid,block>>>(d_matrix);
    cudaDeviceSynchronize();
    printf("warming up using time %lf us \n",(seconds() - start) 
    / 1e6);
    
    start = seconds();
    mathKernal2<<<grid,block>>>(d_matrix);
    cudaDeviceSynchronize();
    printf("mathKernal2 using time %lf us \n",(seconds() - start) 
    / 1e6);

        
    start = seconds();
    mathKernal3<<<grid,block>>>(d_matrix);
    cudaDeviceSynchronize();
    printf("mathKernal3 using time %lf us \n",(seconds() - start) 
    / 1e6);

        
    start = seconds();
    mathKernal4<<<grid,block>>>(d_matrix);
    cudaDeviceSynchronize();
    printf("mathKernal4 using time %lf us \n",(seconds() - start) 
    / 1e6);

    cudaFree(d_matrix);
    free(h_matrix);
    cudaDeviceReset();
    return 0;
}