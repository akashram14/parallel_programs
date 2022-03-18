#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define VECTOR_SIZE 1024

const char* saxpy_kernel = ""
"__kernel void saxpy_kernel(float alpha, __global float *A, __global float *B, __global float *C ){"
"    int id = get_global_id(0);"
"    C[id] = alpha*A[id] + B[id];}"
";";


int main(int argc, char **argv){

    
    float alpha = 2.0;
    float *A = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    float *B = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    float *C = (float*)malloc(sizeof(float)*VECTOR_SIZE);

    for(int i=0; i<VECTOR_SIZE; i++){
        A[i] = i;
        B[i] = VECTOR_SIZE - i;
        C[i] = 0;
    }

    cl_uint numplatforms;
    cl_platform_id* platforms = NULL;
    cl_int clStatus = clGetPlatformIDs(0, NULL, &numplatforms);
    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*numplatforms);


    clStatus = clGetPlatformIDs(numplatforms, platforms, NULL);


    cl_device_id* device_list = NULL;
    cl_uint num_devices;
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    device_list = (cl_device_id*)malloc(sizeof(cl_device_id)*num_devices);
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);


    cl_context context;
    context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &clStatus);

    cl_command_queue command_queue;
    command_queue = clCreateCommandQueueWithProperties(context, device_list[0], NULL, &clStatus);


    cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*VECTOR_SIZE, NULL, &clStatus);
    cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*VECTOR_SIZE, NULL, &clStatus);
    cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*VECTOR_SIZE, NULL, &clStatus);


    clStatus = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, sizeof(float)*VECTOR_SIZE, A, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0, sizeof(float)*VECTOR_SIZE, B, 0, NULL, NULL);


    cl_program program;
    program = clCreateProgramWithSource(context, 1, (const char **)&saxpy_kernel, NULL, &clStatus);

    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "saxpy_kernel", &clStatus);

    clStatus = clSetKernelArg(kernel, 0, sizeof(float), (void*)&alpha);
    clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem),(void*)&A_clmem);
    clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem),(void*)&B_clmem);
    clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem),(void*)&C_clmem);


    size_t global_work_size = VECTOR_SIZE;
    size_t local_work_size = 64;

    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

    clStatus = clEnqueueReadBuffer(command_queue, C_clmem, CL_TRUE, 0, sizeof(float)*VECTOR_SIZE, C, 0, NULL, NULL);

    clStatus = clFlush(command_queue);
    clStatus = clFinish(command_queue);

    // Display the result to the screen
    for (int i = 0; i < VECTOR_SIZE; i++)
        printf("%f * %f + %f = %f\n", alpha, A[i], B[i], C[i]);

    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(A_clmem);
    clStatus = clReleaseMemObject(B_clmem);
    clStatus = clReleaseMemObject(C_clmem);
    clStatus = clReleaseCommandQueue(command_queue);
    clStatus = clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    free(platforms);
    free(device_list);





}