#include "cuda_kernal.h"

long seconds(){
    timeval t;
    gettimeofday(&t,NULL);
    return t.tv_sec * 1e6 + t.tv_usec;
}

void complex_fft_test(cuComplex * input,int paral,int input_size){
    cuComplex * d_input;
    cudaMalloc(&d_input,sizeof(cuComplex) * input_size * paral);
    cudaMemcpy(d_input,input,sizeof(cuComplex) * input_size * paral,cudaMemcpyHostToDevice);

    cuComplex * d_output,* h_output;
    h_output = (cuComplex *)malloc(sizeof(cuComplex) * input_size * paral);
    cudaMalloc(&d_output,sizeof(cuComplex) * input_size * paral);

    cudaEvent_t start,end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cufftHandle handle;
    cufftCreate(&handle);

    // 动目标检测的FFT变换
    int c_n[1] = {input_size};
    int c_inembed_n[2] = {input_size,paral};
    int c_onembed_n[2] = {input_size,paral};
    cufftPlanMany(&handle,1,c_n,
        c_inembed_n,1,input_size,
        c_onembed_n,1,input_size,CUFFT_C2C,paral);
    
    cudaEventRecord(start);
    cufftExecC2C(handle,d_input,d_output,CUFFT_FORWARD);
    cudaEventRecord(end);

    cudaEventSynchronize(end);
    float times;
    cudaEventElapsedTime(&times,start,end);
    printf("[%4d,%4d]complex fft elapsed time %f us\n",input_size,paral,times * 1000);

    // cudaMemcpy(h_output,d_output,sizeof(cuComplex) * input_size *  paral,cudaMemcpyDeviceToHost);
    
    // for(int i=0;i<input_size * paral;i++){
    //     printf("(%f ,%fi) ",h_output[i].x,h_output[i].y);
    // }
    // printf("\n");
    // printf("(%f ,%fi) ",h_output[input_size * paral - 1].x,h_output[input_size * paral - 1].y);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);
    cufftDestroy(handle);
}
__global__ void test_array(float ** d_input){
    printf("you input : \n");
    for(int i=0;i<9;i++){
        printf("%f ",d_input[0][i]);
    }
    printf("\n end you input : \n");
}

__global__ void change(float ** d_input,float * d_input_all,int input_size,int paral){
    for(int i=0;i<paral;i++){
        d_input[i] = d_input_all + i * input_size;
    }
}

void float_invert(float * input,int input_size,int paral){
    int size = sqrt(input_size);
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 此处不能直接使用指针数组，这个是分配在主机内存无法在设备上调用，需要手动在设备上进行分配
    float * d_input_all,*d_output_all;
    float ** d_input,**d_output;
    CHECK(cudaMalloc(&d_input,sizeof(float*) * paral));
    CHECK(cudaMalloc(&d_output,sizeof(float*) * paral));
    CHECK(cudaMalloc(&d_input_all,sizeof(float) * input_size * paral));
    CHECK(cudaMemcpy(d_input_all,input,sizeof(float) * input_size * paral,cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_output_all,sizeof(float) * input_size * paral));
    
    // for(int i=0;i<paral;i++){
    //     d_input[i] = d_input_all + i * input_size;
    //     d_output[i] = d_output_all + i * input_size;
    // }
    change<<<1,1>>>(d_input,d_input_all,input_size,paral);
    change<<<1,1>>>(d_output,d_output_all,input_size,paral);
    test_array<<<1,1>>>(d_input);
    cudaDeviceSynchronize();
    int * info;
    cudaMalloc((void **)&info,sizeof(int) * paral);
    int * h_info;
    h_info = (int *)malloc(sizeof(int) * paral);

    // int * pivotArray;
    // cudaMalloc((void **)&pivotArray,sizeof(int) * size * paral);

    long start = seconds();
    // cublasSgetrfBatched(handle,size,d_input,size,pivotArray,info,paral);
        
    // cublasSgetriBatched(handle,size,d_input,size,pivotArray,d_output,size,info,paral);

    // cudaEventRecord(start);
    
    CHECK_STATUS(cublasSmatinvBatched(handle,size,d_input,size,d_output,size,info,paral));

    float *h_output;
    h_output = (float *) malloc(sizeof(float) * input_size * paral);
    cudaMemcpy(h_output,d_output_all,sizeof(float) * input_size * paral,cudaMemcpyDeviceToHost);

    for(int x = 0;x<paral;x++){
        for(int i=0;i<size;i++){
            for(int j=0;j<size;j++){
                printf("%f ",h_output[i* size + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    
   
    printf("[%4d x %4d] x [%3d]complex invert elapsed time %ld us\n",size,size,paral,(seconds() - start));

    free(h_info);
    cudaFree(info);
    cudaFree(d_input);
    cudaFree(d_output);
    cublasDestroy(handle);
}

int check_info(int * info,int paral){
    for(int i=0;i<paral;i++){
        if(info[i] != 0){
            printf("info calculate failed !\n");
            return -1;
        }
    }
    return 0;
}

void print_result_invert(int paral,cuComplex * h_output,int size){
    for(int i=0;i<paral * size * size;i++){
        if(i % (size * size) == 0){
            printf("the %d array: \n",i / (size * size));
        }
        if(i % (size * size) % size == 0){
            printf("\n");
        }
        printf("(%f,%fi) ",h_output[i].x,h_output[i].y);
    }
    printf("\n");
}

void complex_invert(cuComplex * input,int input_size,int paral){
    int size = sqrt(input_size);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    cudaEvent_t start,end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cuComplex * d_input_data, * d_output_data;
    CHECK(cudaMalloc(&d_input_data,sizeof(cuComplex) * input_size * paral));
    CHECK(cudaMemcpy(d_input_data,input,sizeof(cuComplex) * input_size * paral,cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&d_output_data,sizeof(cuComplex) * input_size * paral));
    CHECK(cudaMemcpy(d_output_data,input,sizeof(cuComplex) * input_size * paral,cudaMemcpyHostToDevice));

    cuComplex * data_address[paral],*output_data_address[paral];
    for(int i=0;i<paral;i++){
        data_address[i] = d_input_data + i * input_size;
        output_data_address[i] = d_output_data + i * input_size;
    }

    cuComplex ** d_input,** d_output;
    CHECK(cudaMalloc(&d_input,sizeof(cuComplex *) * paral));
    CHECK(cudaMalloc(&d_output,sizeof(cuComplex *) * paral));

    CHECK(cudaMemcpy(d_input,data_address,sizeof(cuComplex *) * paral,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_output,output_data_address,sizeof(cuComplex *) * paral,cudaMemcpyHostToDevice));
    // printf("at ehre 154\n");
    // for(int i=0;i<paral;i++){
    //     CHECK_STATUS(cublasSetMatrix(size,size,sizeof(cuComplex),input + i * input_size,size,d_input[i],size));
    //     CHECK_STATUS(cublasSetMatrix(size,size,sizeof(cuComplex),input + i * input_size,size,d_output[i],size));
    // }
    // printf("at ehre 159\n");
    int * info,* h_info;
    CHECK(cudaMalloc(&info,sizeof(int) * paral));
    h_info = (int *)malloc(sizeof(int) * paral);
    
    cudaEventRecord(start);
    CHECK_STATUS(cublasCmatinvBatched(handle,size,d_input,size,d_output,size,info,paral));
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float times;
    cudaEventElapsedTime(&times,start,end);
    printf("[%4d x %4d] x [%3d]complex invert elapsed time %f us ",size,size,paral,times * 1000);

    cudaMemcpy(h_info,info,sizeof(int) * paral,cudaMemcpyDeviceToHost);
    if(check_info(h_info,paral) >= 0){
        printf("\t calculate success !\n");
    }

    // cuComplex * h_output;
    // h_output =(cuComplex *) malloc(sizeof(cuComplex ) * paral * input_size);

    // cudaMemcpy(h_output,d_output_data,sizeof(cuComplex) * input_size * paral,cudaMemcpyDeviceToHost);

    // print_result_invert(paral,h_output,size);

    cudaFree(info);
    free(h_info);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_input_data);
    cudaFree(d_output_data);


    cudaEventDestroy(start);
    cudaEventDestroy(end);
    // free(h_output);
    cublasDestroy(handle);
}

void complex_matrix_gemm(cuComplex * matrix_a,cuComplex * matrix_b,cuComplex * matrix_c,int m,int n,int k){
    cudaEvent_t start,end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    cuComplex *d_matrix_a,*d_matrix_b,*d_matrix_c;

    cudaMalloc((void **)&d_matrix_a,sizeof(cuComplex) * m * n);
    cudaMalloc((void **)&d_matrix_b,sizeof(cuComplex) * n * k);
    cudaMalloc((void **)&d_matrix_c,sizeof(cuComplex) * m * k);

    CHECK_STATUS(cublasSetMatrix(m,n,sizeof(cuComplex),matrix_a,m,d_matrix_a,m));
    CHECK_STATUS(cublasSetMatrix(n,k,sizeof(cuComplex),matrix_b,n,d_matrix_b,n));
    cuComplex alpha = make_cuComplex(1,0);
    cuComplex belta = make_cuComplex(0,0);
    cudaEventRecord(start);
    // CHECK_STATUS(cublasCgemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,d_matrix_a,CUDA_C_32F,m,d_matrix_b,CUDA_C_32F,n,&belta,d_matrix_c,CUDA_C_32F,m));
    // CHECK_STATUS(cublasCgemm3m(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,d_matrix_a,m,d_matrix_b,n,&belta,d_matrix_c,m));
    CHECK_STATUS(cublasCgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,d_matrix_a,m,d_matrix_b,n,&belta,d_matrix_c,m));
    cudaEventRecord(end);

    cudaEventSynchronize(end);
    float times;
    cudaEventElapsedTime(&times,start,end);
    printf("[%4d,%4d] x [%4d,%4d] complex elapsed time %f us\n",m,n,n,k,times * 1000);

    // cudaMemcpy(matrix_c,d_matrix_c,sizeof(cuComplex)*m*k,cudaMemcpyDeviceToHost);
    // CHECK_STATUS(cublasGetMatrix(m,k,sizeof(cuComplex),d_matrix_c,m,matrix_c,m));
    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_matrix_c);


    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cublasDestroy(handle);

}

void complex_matrix_transpose(cuComplex * input,int width,int height){

    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cuComplex * d_input,* d_output;
    CHECK(cudaMalloc((void **)&d_input,sizeof(cuComplex) * width * height));
    CHECK(cudaMalloc((void **)&d_output,sizeof(cuComplex) * width * height));

    CHECK(cudaMemcpy(d_input,input,sizeof(cuComplex) * width * height,cudaMemcpyHostToDevice));
    // cuComplex ** h_input_address,*h_output_data,**h_output_address;
    // h_input_address = (cuComplex **)malloc(sizeof(cuComplex *) * input_size);
    // h_output_address = (cuComplex **)malloc(sizeof(cuComplex *) * input_size);
    // h_output_data =(cuComplex *) malloc(sizeof(cuComplex) * input_size);
    // memset(h_output_data,0,sizeof(cuComplex) * input_size);

    // for(int i=0;i<size;i++){
    //     h_input_address[i] = input + size * i;
    //     h_output_address[i] = h_output_data + size * i;
    // }

    // cuComplex * d_input_data,** d_input_address;
    // cuComplex * d_output_data,** d_output_address;

    // CHECK(cudaMalloc((void **)&d_input_data,sizeof(cuComplex) * input_size));
    // CHECK(cudaMalloc((void **)&d_input_address,sizeof(cuComplex *) * input_size));

    // CHECK(cudaMalloc((void **)&d_output_data,sizeof(cuComplex) * input_size));
    // CHECK(cudaMalloc((void **)&d_output_address,sizeof(cuComplex *) * input_size));

    // CHECK(cudaMemcpy(d_input_data,input,sizeof(cuComplex) * input_size,cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_input_address,h_input_address,sizeof(cuComplex *) * size,cudaMemcpyHostToDevice));

    // CHECK(cudaMemcpy(d_output_data,h_output_data,sizeof(cuComplex) * input_size,cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_output_address,h_output_address,sizeof(cuComplex *) * size,cudaMemcpyHostToDevice));

    cuComplex alpha = make_cuComplex(1,0);
    cuComplex belta = make_cuComplex(0,0);

    cudaEventRecord(start);
    // (h x w)T  + w x h = w x h
    CHECK_STATUS(cublasCgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,height,width,&alpha,d_input,width,&belta,d_input,height,d_output,height));
    cudaEventRecord(end);

    cudaEventSynchronize(end);
    float times;
    cudaEventElapsedTime(&times,start,end);
    printf("[%5d x %5d]complex transpose elapsed time %f us\n",height,width,times * 1000);

    // cuComplex * h_output;
    // h_output = (cuComplex *)malloc(sizeof(cuComplex) * width * height);
    // cudaMemcpy(h_output,d_output,sizeof(cuComplex) * width * height,cudaMemcpyDeviceToHost);
    // for(int i=0;i<width;i++){
    //     for(int j=0;j<height;j++){
    //         printf("(%4.2f,%4.2fi) ",h_output[i * height + j].x,h_output[i * height + j].y);
    //     }
    //     printf("\n");
    // }
    
    // free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cublasDestroy(handle);   
}

__global__ void multi_kernal(cuComplex * signal,cuComplex * coefficient,cuComplex * output,int width,int height){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if(idx < width && idy < height){
        int target_id = idx + idy * width;
        output[target_id].x = signal[target_id].x * coefficient[idx].x - signal[target_id].y * coefficient[idx].y;
        output[target_id].y = signal[target_id].x * coefficient[idx].y + signal[target_id].y * coefficient[idx].x;
    }
}

__global__ void multi_kernal_col(cuComplex * signal,cuComplex * coefficient,cuComplex * output,int width,int height){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    if(idx < width && idy < height){
        int target_id = idx + idy * width;
        output[target_id].x = signal[target_id].x * coefficient[idy].x - signal[target_id].y * coefficient[idy].y;
        output[target_id].y = signal[target_id].x * coefficient[idy].y + signal[target_id].y * coefficient[idy].x;
    }
}
__global__ void multi_kernal_4(cuComplex * signal,cuComplex * coefficient,cuComplex * output,int width,int height){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int expend_idx = idx * 4;
    if(idx < width && idy < height){
        int target_id = idx * 4 + idy * width;
        output[target_id].x = signal[target_id].x * coefficient[expend_idx].x - signal[target_id].y * coefficient[expend_idx].y;
        output[target_id].y = signal[target_id].x * coefficient[expend_idx].y + signal[target_id].y * coefficient[expend_idx].x;

        output[target_id + 1].x = signal[target_id + 1].x * coefficient[expend_idx + 1].x - signal[target_id + 1].y * coefficient[expend_idx + 1].y;
        output[target_id + 1].y = signal[target_id + 1].x * coefficient[expend_idx + 1].y + signal[target_id + 1].y * coefficient[expend_idx + 1].x;

        output[target_id + 2].x = signal[target_id + 2].x * coefficient[expend_idx + 2].x - signal[target_id + 2].y * coefficient[expend_idx + 2].y;
        output[target_id + 2].y = signal[target_id + 2].x * coefficient[expend_idx + 2].y + signal[target_id + 2].y * coefficient[expend_idx + 2].x;

        output[target_id + 3].x = signal[target_id + 3].x * coefficient[expend_idx + 3].x - signal[target_id + 3].y * coefficient[expend_idx + 3].y;
        output[target_id + 3].y = signal[target_id + 3].x * coefficient[expend_idx + 3].y + signal[target_id + 3].y * coefficient[expend_idx + 3].x;
    }
}

/**
 * ifft(fft(sginal) * fft(coefficient))
 * row * col
*/
float match_filter(cuComplex * input_signal,cuComplex * coefficient,int input_size,int paral,int dimx,int dimy){
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float times = 0.0;
    cuComplex * d_input_signal,* d_coefficient,* d_output;
    size_t data_bytes = sizeof(cuComplex) * input_size * paral,coefficient_bytes = sizeof(cuComplex) * input_size * paral;
    CHECK(cudaMalloc((void **)&d_input_signal,data_bytes));
    CHECK(cudaMalloc((void **)&d_output,data_bytes));
    CHECK(cudaMalloc((void **)&d_coefficient,coefficient_bytes));

    CHECK(cudaMemcpy(d_input_signal,input_signal,data_bytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_coefficient,coefficient,data_bytes,cudaMemcpyHostToDevice));
    cufftHandle handle_signal,hanlde_coefficient;
    cufftCreate(&handle_signal);
    cufftCreate(&hanlde_coefficient);


    /************fft(coefficient)*************/
    int n_c[1] = {input_size};
    int inembed_c[2] = {input_size,paral};
    int onembed_c[2] = {input_size,paral};
    cufftPlanMany(&hanlde_coefficient,1,n_c,inembed_c,1,input_size,onembed_c,1,input_size,CUFFT_C2C,paral);

    
    /********************end******************/
    /************fft(signal)*************/
    int n[1] = {input_size};
    int inembed[2] = {input_size,paral};
    int onembed[2] = {input_size,paral};
    cufftPlanMany(&handle_signal,1,n,inembed,1,input_size,onembed,1,input_size,CUFFT_C2C,paral);
    /**************end******************/

    dim3 block(dimx,dimy),grid((input_size + block.x - 1)/block.x,(paral + block.y -1)/block.y);
    // block.x /= 4;

    cudaEventRecord(start);
    cufftExecC2C(hanlde_coefficient,d_coefficient,d_coefficient,CUFFT_FORWARD); // fft(co)
    cufftExecC2C(handle_signal,d_input_signal,d_input_signal,CUFFT_FORWARD); // fft(signal)
    multi_kernal<<<grid,block>>>(d_input_signal,d_coefficient,d_output,input_size,paral); // fft(co) * fft(signal)
    
    // multi_kernal_4<<<grid,block>>>(d_input_signal,d_coefficient,d_output,input_size,paral); // fft(co) * fft(signal)
    cufftExecC2C(handle_signal,d_output,d_output,CUFFT_INVERSE); // ifft(fft(co) * fft(signal))
    cudaEventRecord(end);

    cudaEventSynchronize(end);

    cudaEventElapsedTime(&times,start,end);
    printf("[%5d x %5d]complex match filter save with rows elapsed time %f us ",input_size,paral,times * 1000);
    printf("\t multi kernal block:[%d,%d,%d],grid:[%d,%d,%d] \n",block.x,block.y,block.z,grid.x,grid.y,grid.z);

    cuComplex * h_output;
    h_output = (cuComplex *)malloc(data_bytes);
    cudaMemcpy(h_output,d_output,data_bytes,cudaMemcpyDeviceToHost);

    for(int i=0;i<input_size * paral;i++){
        printf("ifft(fft(%5.2f,%5.2fi) x fft(%5.2f,%5.2fi)) = (%5.2f,%5.2fi) \n",input_signal[i].x,input_signal[i].y,coefficient[i].x,coefficient[i].y,h_output[i].x,h_output[i].y);
    }
    
    free(h_output);
    cudaFree(d_coefficient);
    cudaFree(d_input_signal);
    cufftDestroy(handle_signal);
    cufftDestroy(hanlde_coefficient);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    // cublasDestroy(blas_handle);
    return times * 1000;
}

float match_filter_rowfft_colkernal(cuComplex * input_signal,cuComplex * coefficient,int input_size,int paral,int dimx,int dimy){
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float times = 0.0;
    cuComplex * d_input_signal,* d_coefficient,* d_output;
    size_t data_bytes = sizeof(cuComplex) * input_size * paral,coefficient_bytes = sizeof(cuComplex) * input_size * paral;
    CHECK(cudaMalloc((void **)&d_input_signal,data_bytes));
    CHECK(cudaMalloc((void **)&d_output,data_bytes));
    CHECK(cudaMalloc((void **)&d_coefficient,coefficient_bytes));

    CHECK(cudaMemcpy(d_input_signal,input_signal,data_bytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_coefficient,coefficient,data_bytes,cudaMemcpyHostToDevice));
    cufftHandle handle_signal,hanlde_coefficient;
    cufftCreate(&handle_signal);
    cufftCreate(&hanlde_coefficient);


    /************fft(coefficient)*************/
    int n_c[1] = {input_size};
    int inembed_c[2] = {input_size,paral};
    int onembed_c[2] = {input_size,paral};
    cufftPlanMany(&hanlde_coefficient,1,n_c,inembed_c,1,input_size,onembed_c,1,input_size,CUFFT_C2C,paral);

    
    /********************end******************/
    /************fft(signal)*************/
    int n[1] = {input_size};
    int inembed[2] = {input_size,paral};
    int onembed[2] = {input_size,paral};
    cufftPlanMany(&handle_signal,1,n,inembed,1,input_size,onembed,1,input_size,CUFFT_C2C,paral);
    /**************end******************/

    /****************transpose**************/
    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);

    cuComplex alpha = make_cuComplex(1,0);
    cuComplex belta = make_cuComplex(0,0);

    /****************transpose end**************/
    dim3 block(dimx,dimy),grid((input_size + block.x - 1)/block.x,(paral + block.y -1)/block.y);
    // block.x /= 4;

    cudaEventRecord(start);
    cufftExecC2C(hanlde_coefficient,d_coefficient,d_coefficient,CUFFT_FORWARD); // fft(co)
    cufftExecC2C(handle_signal,d_input_signal,d_input_signal,CUFFT_FORWARD); // fft(signal)
    CHECK_STATUS(cublasCgeam(blas_handle,CUBLAS_OP_T,CUBLAS_OP_N,paral,input_size,&alpha,d_input_signal,input_size,&belta,d_input_signal,paral,d_output,paral));
    // multi_kernal<<<grid,block>>>(d_input_signal,d_coefficient,d_output,input_size,paral); // fft(co) * fft(signal)
    multi_kernal_col<<<grid,block>>>(d_output,d_coefficient,d_input_signal,paral,input_size); // fft(co) * fft(signal)
    CHECK_STATUS(cublasCgeam(blas_handle,CUBLAS_OP_T,CUBLAS_OP_N,input_size,paral,&alpha,d_input_signal,paral,&belta,d_input_signal,input_size,d_output,input_size));
    // multi_kernal_4<<<grid,block>>>(d_input_signal,d_coefficient,d_output,input_size,paral); // fft(co) * fft(signal)
    cufftExecC2C(handle_signal,d_output,d_output,CUFFT_INVERSE); // ifft(fft(co) * fft(signal))
    cudaEventRecord(end);

    cudaEventSynchronize(end);

    cudaEventElapsedTime(&times,start,end);
    printf("[%5d x %5d]complex match filter save with rows elapsed time %f us ",input_size,paral,times * 1000);
    printf("\t multi kernal block:[%d,%d,%d],grid:[%d,%d,%d] \n",block.x,block.y,block.z,grid.x,grid.y,grid.z);

    cuComplex * h_output;
    h_output = (cuComplex *)malloc(data_bytes);
    cudaMemcpy(h_output,d_output,data_bytes,cudaMemcpyDeviceToHost);

    for(int i=0;i<input_size * paral;i++){
        printf("ifft(fft(%5.2f,%5.2fi) x fft(%5.2f,%5.2fi)) = (%5.2f,%5.2fi) \n",input_signal[i].x,input_signal[i].y,coefficient[i].x,coefficient[i].y,h_output[i].x,h_output[i].y);
    }
    
    free(h_output);
    cudaFree(d_coefficient);
    cudaFree(d_input_signal);
    cufftDestroy(handle_signal);
    cufftDestroy(hanlde_coefficient);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cublasDestroy(blas_handle);
    // cublasDestroy(blas_handle);
    return times * 1000;
}

float match_filter_col(cuComplex * input_signal,cuComplex * coefficient,int input_size,int paral,int dimx,int dimy){
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float times = 0.0;
    cuComplex * d_input_signal,* d_coefficient,* d_output;
    size_t data_bytes = sizeof(cuComplex) * input_size * paral,coefficient_bytes = sizeof(cuComplex) * input_size * paral;
    CHECK(cudaMalloc((void **)&d_input_signal,data_bytes));
    CHECK(cudaMalloc((void **)&d_output,data_bytes));
    CHECK(cudaMalloc((void **)&d_coefficient,coefficient_bytes));

    CHECK(cudaMemcpy(d_input_signal,input_signal,data_bytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_coefficient,coefficient,data_bytes,cudaMemcpyHostToDevice));
    cufftHandle handle_signal,hanlde_coefficient;
    cufftCreate(&handle_signal);
    cufftCreate(&hanlde_coefficient);


    /************fft(coefficient)*************/
    int n_c[1] = {input_size};
    int inembed_c[2] = {input_size,paral};
    int onembed_c[2] = {input_size,paral};
    cufftPlanMany(&hanlde_coefficient,1,n_c,inembed_c,1,input_size,onembed_c,1,input_size,CUFFT_C2C,paral);
    
    /********************end******************/
    /************fft(signal)*************/
    int n[1] = {input_size};
    int inembed[2] = {paral,input_size};
    int onembed[2] = {paral,input_size};
    if(paral == 1){
        cufftPlanMany(&handle_signal,1,n,inembed,1,input_size,onembed,1,input_size,CUFFT_C2C,paral);
    }else{
        cufftPlanMany(&handle_signal,1,n,inembed,paral,1,onembed,paral,1,CUFFT_C2C,paral);
    }
    
    /**************end******************/

    dim3 block(dimx,dimy),grid((input_size + block.x - 1)/block.x,(paral + block.y -1)/block.y);
    

    cudaEventRecord(start);
    cufftExecC2C(hanlde_coefficient,d_coefficient,d_coefficient,CUFFT_FORWARD); // fft(co)
    cufftExecC2C(handle_signal,d_input_signal,d_input_signal,CUFFT_FORWARD); // fft(signal)
    multi_kernal_col<<<grid,block>>>(d_input_signal,d_coefficient,d_output,paral,input_size); // fft(co) * fft(signal)
    // block.x /= 4;
    // multi_kernal_4<<<grid,block>>>(d_input_signal,d_coefficient,d_output,input_size,paral); // fft(co) * fft(signal)
    cufftExecC2C(handle_signal,d_output,d_output,CUFFT_INVERSE); // ifft(fft(co) * fft(signal))
    cudaEventRecord(end);

    cudaEventSynchronize(end);

    cudaEventElapsedTime(&times,start,end);
    printf("[%5d x %5d]complex match filter elapsed time %f us ",input_size,paral,times * 1000);
    printf("\t multi kernal block:[%d,%d,%d],grid:[%d,%d,%d] \n",block.x,block.y,block.z,grid.x,grid.y,grid.z);

    // cuComplex * h_output;
    // h_output = (cuComplex *)malloc(data_bytes);
    // cudaMemcpy(h_output,d_output,data_bytes,cudaMemcpyDeviceToHost);

    // for(int i=0;i<input_size * paral;i++){
    //     printf("ifft(fft(%5.2f,%5.2fi) x fft(%5.2f,%5.2fi)) = (%5.2f,%5.2fi) \n",input_signal[i].x,input_signal[i].y,coefficient[i / paral].x,coefficient[i / paral].y,h_output[i].x,h_output[i].y);
    // }
    
    // free(h_output);
    cudaFree(d_coefficient);
    cudaFree(d_input_signal);
    cufftDestroy(handle_signal);
    cufftDestroy(hanlde_coefficient);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    // cublasDestroy(blas_handle);
    return times * 1000;
}


/**
 * ifft(fft(sginal) * fft(coefficient))
*/
float match_filter_streams(cuComplex * input_signal,cuComplex * coefficient,int input_size,int paral,int dimx,int dimy){
    cudaStream_t streams[3];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    cudaStreamCreate(&streams[2]);

    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float times = 0.0;
    cuComplex * d_input_signal,* d_coefficient,* d_output;
    size_t data_bytes = sizeof(cuComplex) * input_size * paral,coefficient_bytes = sizeof(cuComplex) * input_size * paral;
    CHECK(cudaMalloc((void **)&d_input_signal,data_bytes));
    CHECK(cudaMalloc((void **)&d_output,data_bytes));
    CHECK(cudaMalloc((void **)&d_coefficient,coefficient_bytes));

    CHECK(cudaMemcpy(d_input_signal,input_signal,data_bytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_coefficient,coefficient,data_bytes,cudaMemcpyHostToDevice));
    cufftHandle handle_signal,hanlde_coefficient;
    cufftCreate(&handle_signal);
    cufftCreate(&hanlde_coefficient);
    /************fft(coefficient)*************/
    int n_c[1] = {input_size};
    int inembed_c[2] = {input_size,paral};
    int onembed_c[2] = {input_size,paral};
    cufftPlanMany(&hanlde_coefficient,1,n_c,inembed_c,1,input_size,onembed_c,1,input_size,CUFFT_C2C,paral);
    cufftSetStream(hanlde_coefficient,streams[0]);
    /********************end******************/
    /************fft(signal)*************/
    int n[1] = {input_size};
    int inembed[2] = {input_size,paral};
    int onembed[2] = {input_size,paral};
    cufftPlanMany(&handle_signal,1,n,inembed,1,input_size,onembed,1,input_size,CUFFT_C2C,paral);
    cufftSetStream(handle_signal,streams[1]);
    /**************end******************/
    /*****************multi-mv***************/
    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    cuComplex alpha = make_cuComplex(1,0);
    cuComplex beta = make_cuComplex(0,0);
    /********************end*****************/

    dim3 block(dimx,dimy),grid((input_size + block.x - 1)/block.x,(paral + block.y -1)/block.y);

    
    cudaEventRecord(start);
    cufftExecC2C(hanlde_coefficient,d_coefficient,d_coefficient,CUFFT_FORWARD); // fft(co)
    cufftExecC2C(handle_signal,d_input_signal,d_input_signal,CUFFT_FORWARD); // fft(signal)
    // CHECK_STATUS(cublasCgemv(blas_handle,CUBLAS_OP_N,paral,input_size,&alpha,d_input_signal,input_size,d_coefficient,1,&beta,d_output,input_size));
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    multi_kernal<<<grid,block,0,streams[2]>>>(d_input_signal,d_coefficient,d_output,input_size,paral); // fft(co) * fft(signal)
    // cudaDeviceSynchronize();
    cudaStreamSynchronize(streams[2]);
    cufftExecC2C(handle_signal,d_output,d_output,CUFFT_INVERSE); // ifft(fft(co) * fft(signal))

    cudaEventRecord(end);

    cudaEventSynchronize(end);

    cudaEventElapsedTime(&times,start,end);
    printf("[%5d x %5d]complex match filter with 3 streams elapsed time %f us ",input_size,paral,times * 1000);
    printf("\t multi kernal block:[%d,%d,%d],grid:[%d,%d,%d] \n",block.x,block.y,block.z,grid.x,grid.y,grid.z);

    // cuComplex * h_output;
    // h_output = (cuComplex *)malloc(data_bytes);
    // cudaMemcpy(h_output,d_output,data_bytes,cudaMemcpyDeviceToHost);

    // for(int i=0;i<input_size * paral;i++){
    //     printf("ifft(fft(%5.2f,%5.2fi) x fft(%5.2f,%5.2fi)) = (%5.2f,%5.2fi) \n",input_signal[i].x,input_signal[i].y,coefficient[i].x,coefficient[i].y,h_output[i].x,h_output[i].y);
    // }
    
    // free(h_output);
    cudaFree(d_coefficient);
    cudaFree(d_input_signal);

    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaStreamDestroy(streams[2]);
    cufftDestroy(handle_signal);
    cufftDestroy(hanlde_coefficient);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    // cublasDestroy(blas_handle);
    return times * 1000;
}

__constant__ cuComplex d_coefficient[8192];

__global__ void conv_kernal(cuComplex * d_input_signal,cuComplex * d_output_signal,int width,int height,int coefficient_size){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int data_idx = idx + idy * width;

    // int data_left_band = idy * width;
    // int data_right_band = (idy + 1) * width;

    int loop = idx > coefficient_size ? coefficient_size:idx;
    cuComplex sum = make_cuComplex(0.0,0.0);
    for(int i=0;i<loop;i++){
        sum.x += d_input_signal[data_idx - i].x * d_coefficient[i].x - d_input_signal[data_idx - i].y * d_coefficient[i].y;
        sum.y += d_input_signal[data_idx - i].x * d_coefficient[i].y + d_input_signal[data_idx - i].y * d_coefficient[i].x;
    }
    d_output_signal[data_idx] = sum;
}

float conv_signal_coeff(cuComplex * input_signal,cuComplex * coefficient,cuComplex * output_signal,int signal_length,int batch,int co_input_size,int dimx,int dimy){
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float times = 0.0;
    int input_size = signal_length * batch;
    cuComplex * d_input_signal,* d_output_signal;
    
    CHECK(cudaMalloc((void **)&d_input_signal,sizeof(cuComplex) * input_size));
    CHECK(cudaMalloc((void **)&d_output_signal,sizeof(cuComplex) * input_size));
    // CHECK(cudaMalloc((void **)&d_coefficient,sizeof(cuComplex) * co_input_size));

    CHECK(cudaMemcpy(d_input_signal,input_signal,sizeof(cuComplex) * input_size,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(d_coefficient,coefficient,sizeof(cuComplex) * co_input_size,0,cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_coefficient,coefficient,sizeof(cuComplex) * co_input_size,cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_output_signal,0,sizeof(cuComplex) * input_size));

    dim3 block(dimx,dimy);
    dim3 grid((signal_length + block.x - 1)/block.x,(batch + block.y - 1) /block.y);
    cudaEventRecord(start);
    conv_kernal<<<grid,block>>>(d_input_signal,d_output_signal,signal_length,batch,co_input_size);
    cudaEventRecord(end);

    cudaEventSynchronize(end);
    cudaEventElapsedTime(&times,start,end);
    printf("[%5d x %5d] cov elapsed time %f us ",signal_length,batch,times * 1000);
    printf("\t multi kernal block:[%d,%d,%d],grid:[%d,%d,%d] \n",block.x,block.y,block.z,grid.x,grid.y,grid.z);

    cudaMemcpy(output_signal,d_output_signal,sizeof(cuComplex) * input_size,cudaMemcpyDeviceToHost);

    cudaFree(d_input_signal);
    // cudaFree(d_coefficient);
    cudaFree(d_output_signal);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return times * 1000;
}

cudaStream_t streams[8];
cudaEvent_t start,end;

void streamAndEvent_init(){
    for(int i=0;i<8;i++){
        cudaStreamCreate(&streams[i]);
    }
    cudaEventCreate(&start);
    cudaEventCreate(&end);
}

void streamAndEvent_Destroy(){
    for(int i=0;i<8;i++){
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(end);
}
void conv_signal_coeff_async(cuComplex * input_signal,cuComplex * coefficient,cuComplex * output_signal,int signal_length,int batch,int co_input_size,int dimx,int dimy,int streamId){
    
    
    float times = 0.0;
    int input_size = signal_length * batch;
    cuComplex * d_input_signal,* d_coefficient,* d_output_signal;
    
    CHECK(cudaMallocAsync((void **)&d_input_signal,sizeof(cuComplex) * input_size,streams[streamId]));
    CHECK(cudaMallocAsync((void **)&d_output_signal,sizeof(cuComplex) * input_size,streams[streamId]));
    CHECK(cudaMallocAsync((void **)&d_coefficient,sizeof(cuComplex) * co_input_size,streams[streamId]));

    CHECK(cudaMemcpyAsync(d_input_signal,input_signal,sizeof(cuComplex) * input_size,cudaMemcpyHostToDevice,streams[streamId]));
    CHECK(cudaMemcpyToSymbolAsync(d_coefficient,coefficient,sizeof(cuComplex) * co_input_size,0,cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpyAsync(d_coefficient,coefficient,sizeof(cuComplex) * co_input_size,cudaMemcpyHostToDevice,streams[streamId]));
    // CHECK(cudaMemset(d_output_signal,0,sizeof(cuComplex) * input_size));

    dim3 block(dimx,dimy);
    dim3 grid((signal_length + block.x - 1)/block.x,(batch + block.y - 1) /block.y);
    cudaEventRecord(start);
    conv_kernal<<<grid,block,0,streams[streamId]>>>(d_input_signal,d_output_signal,signal_length,batch,co_input_size);
    cudaEventRecord(end);

    cudaEventSynchronize(end);
    cudaEventElapsedTime(&times,start,end);
    printf("[%5d x %5d] cov elapsed time %f us ",signal_length,batch,times * 1000);
    printf("\t multi kernal block:[%d,%d,%d],grid:[%d,%d,%d] \n",block.x,block.y,block.z,grid.x,grid.y,grid.z);

    cudaMemcpyAsync(output_signal,d_output_signal,sizeof(cuComplex) * input_size,cudaMemcpyDeviceToHost,streams[streamId]);

    cudaFreeAsync(d_input_signal,streams[streamId]);
    cudaFreeAsync(d_coefficient,streams[streamId]);
    cudaFreeAsync(d_output_signal,streams[streamId]);

}