// This program implements a vector addition using OpenCL

// System includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <cstring>
// OpenCL includes
#include <CL/cl.h>
// CNN
#include "cnn.h"

using namespace std;
#define O true
#define X false
const bool tbl[6][16] = {
	O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
	O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
	O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
	X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
	X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
	X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
};
#undef O
#undef X
//全局变量声明
cl_int status;
cl_uint numPlatforms;
cl_platform_id *platforms;
cl_uint numDevices;
cl_device_id *devices;
cl_context context;
cl_command_queue cmdQueue;
cl_program program1;//kernel.cl
cl_program program2;//backward.cl
cl_program program3;//update_delta.cl
//Forward_C1()
cl_kernel kernel1;
cl_mem clbuf_data_single_image;  // Input array on the device
cl_mem clbuf_weight_C1;  // Input array on the device
cl_mem clbuf_bias_C1;  // Input array on the device
cl_mem clbuf_neuron_C1; // Output array on the device
//Forward_S2()
cl_kernel kernel2;
cl_mem clbuf_weight_S2;  // Input array on the device
cl_mem clbuf_bias_S2;  // Input array on the device
cl_mem clbuf_neuron_S2; // Output array on the device
//Forward_C3()
cl_kernel kernel3;
cl_mem clbuf_weight_C3;  // Input array on the device
cl_mem clbuf_bias_C3;  // Input array on the device
cl_mem clbuf_neuron_C3; // Output array on the device
cl_mem clbuf_tbl;
//Forward_S4()
cl_kernel kernel4;
cl_mem clbuf_weight_S4;  // Input array on the device
cl_mem clbuf_bias_S4;  // Input array on the device
cl_mem clbuf_neuron_S4; // Output array on the device
//Forward_C5()
cl_kernel kernel5;
cl_mem clbuf_weight_C5;  // Input array on the device
cl_mem clbuf_bias_C5;  // Input array on the device
cl_mem clbuf_neuron_C5; // Output array on the device
//Forward_output()
cl_kernel kernel6;
cl_mem clbuf_weight_output;  // Input array on the device
cl_mem clbuf_bias_output;  // Input array on the device
cl_mem clbuf_neuron_output; // Output array on the device
//Backward_output()
cl_kernel kernel7; 
cl_mem clbuf_delta_neuron_output;
cl_mem clbuf_data_single_label; 
//Backward_C5()
cl_kernel kernel8; 
cl_mem clbuf_delta_neuron_C5; 
cl_mem clbuf_delta_weight_output; 
//Backward_S4()
cl_kernel kernel9; 
cl_mem clbuf_delta_neuron_S4; 
cl_mem clbuf_delta_weight_C5; 
//Backward_C3()
cl_kernel kernel10; 
cl_mem clbuf_delta_neuron_C3; 
cl_mem clbuf_delta_weight_S4;
//Backward_S2()
cl_kernel kernel11; 
cl_kernel kernel12;
cl_mem clbuf_delta_neuron_S2; 
cl_mem clbuf_delta_weight_C3; 
//Backward_C1()
cl_kernel kernel13; 
cl_mem clbuf_delta_neuron_C1; 
cl_mem clbuf_delta_weight_S2;
//Backward_input()
cl_kernel kernel14;
cl_kernel kernel15;
cl_mem clbuf_delta_neuron_input; 
cl_mem clbuf_delta_weight_C1;
//DeltaBias()
cl_kernel kernel16;
cl_mem clbuf_delta_bias_output;
cl_mem clbuf_delta_bias_C5;
cl_mem clbuf_delta_bias_S4;
cl_mem clbuf_delta_bias_C3;
cl_mem clbuf_delta_bias_S2;
cl_mem clbuf_delta_bias_C1;
//UpdateWeights();
cl_kernel kernel17;
cl_mem clbuf_E_weight_output;
cl_mem clbuf_E_weight_C5;
cl_mem clbuf_E_weight_S4;
cl_mem clbuf_E_weight_C3;
cl_mem clbuf_E_weight_S2;
cl_mem clbuf_E_weight_C1;

cl_mem clbuf_E_bias_output;
cl_mem clbuf_E_bias_C5;
cl_mem clbuf_E_bias_S4;
cl_mem clbuf_E_bias_C3;
cl_mem clbuf_E_bias_S2;
cl_mem clbuf_E_bias_C1;

//double equal judge
int dcmp(double x){
    if(fabs(x) < eps) return 0;
    return x<0?-1:1;
}

//read kernel file
int convertToString(const char *filename, std::string& s)
{
    size_t size;
    char* str;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));
    if (f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size + 1];
        if (!str)
        {
            f.close();
            return 0;
        }
        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        s = str;
        delete[] str;
        return 0;
    }
    printf("Error: Failed to open file %s\n", filename);
    return 1;
}

bool CNN::OpenCL_init(){
    //-----------------------------------------------------
    // STEP 1: Discover and initialize the platforms
    //-----------------------------------------------------
    numPlatforms = 0;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if(status) printf("Initialize platforms error!\n");
    //-----------------------------------------------------
    // STEP 2: Discover and initialize the devices
    //----------------------------------------------------- 
    numDevices = 0;
    status = clGetDeviceIDs(platforms[0],CL_DEVICE_TYPE_ALL,0,NULL,&numDevices);
    devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
    status = clGetDeviceIDs(platforms[0],CL_DEVICE_TYPE_ALL,numDevices,devices,NULL);
    if(status) printf("Initialize devices erroe!\n");
    //-----------------------------------------------------
    // STEP 3: Create a context
    //----------------------------------------------------- 
    context = clCreateContext(NULL, numDevices,devices, NULL,NULL,&status);
    //-----------------------------------------------------
    // STEP 4: Create a command queue
    //----------------------------------------------------- 
    cmdQueue = clCreateCommandQueue(context, devices[0],CL_QUEUE_PROFILING_ENABLE,&status);
    //-----------------------------------------------------
    // STEP 5: Create device buffers
    //----------------------------------------------------- 
    //Forward_C1()
    clbuf_data_single_image = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*num_neuron_input_CNN, NULL,&status);//32*32
	clbuf_weight_C1 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_weight_C1_CNN, weight_C1,&status);//6@5*5
    clbuf_bias_C1 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_bias_C1_CNN, bias_C1, &status);//6
	clEnqueueWriteBuffer(cmdQueue, clbuf_weight_C1, CL_FALSE,0,sizeof(float)*len_weight_C1_CNN,weight_C1,0,NULL,NULL);	
    clEnqueueWriteBuffer(cmdQueue, clbuf_bias_C1, CL_FALSE,0,sizeof(float)*len_bias_output_CNN, bias_C1, 0, NULL, NULL);
    clbuf_neuron_C1 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*num_neuron_C1_CNN, NULL, &status);//6@28*28
	//Forward_S2()
	clbuf_weight_S2 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_weight_S2_CNN, weight_S2,&status);//6@1(ave=1)
    clbuf_bias_S2 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_bias_S2_CNN, bias_S2, &status);//6
	clbuf_neuron_S2 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*num_neuron_S2_CNN, NULL, &status);//6@14*14
	clEnqueueWriteBuffer(cmdQueue, clbuf_weight_S2, CL_FALSE,0,sizeof(float)*len_weight_S2_CNN,weight_S2,0,NULL,NULL);
	clEnqueueWriteBuffer(cmdQueue, clbuf_bias_S2, CL_FALSE,0,sizeof(float)*len_bias_S2_CNN,bias_S2,0,NULL,NULL);	
    //Forward_C3()
	clbuf_weight_C3 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_weight_C3_CNN, weight_C3,&status);//16@5*5 × 6(S2)
	clEnqueueWriteBuffer(cmdQueue, clbuf_weight_C3, CL_FALSE,0,sizeof(float)*len_weight_C3_CNN,weight_C3,0,NULL,NULL);
    clbuf_bias_C3 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_bias_C3_CNN, bias_C3, &status);//16
	clEnqueueWriteBuffer(cmdQueue, clbuf_bias_C3, CL_FALSE,0,sizeof(float)*len_bias_C3_CNN,bias_C3,0,NULL,NULL);
    clbuf_neuron_C3 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*num_neuron_C3_CNN, NULL, &status);//16@10*10
	clbuf_tbl = clCreateBuffer(context, CL_MEM_READ_ONLY,sizeof(bool)*6*16, NULL, &status);//bool [6][16]
	clEnqueueWriteBuffer(cmdQueue, clbuf_tbl, CL_FALSE,0,sizeof(bool)*6*16,tbl,0,NULL,NULL);
	//Forward_S4()
	clbuf_weight_S4 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_weight_S4_CNN, weight_S4,&status);//16@1(ave=1)
	clEnqueueWriteBuffer(cmdQueue, clbuf_weight_S4, CL_FALSE,0,sizeof(float)*len_weight_S4_CNN,weight_S4,0,NULL,NULL);
    clbuf_bias_S4 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_bias_S4_CNN, bias_S4, &status);//16
	clEnqueueWriteBuffer(cmdQueue, clbuf_bias_S4, CL_FALSE,0,sizeof(float)*len_bias_S4_CNN, bias_S4,0,NULL,NULL);
    clbuf_neuron_S4 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*num_neuron_S4_CNN, NULL, &status);//16@5*5
	//Forward_C5()
	clbuf_weight_C5 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_weight_C5_CNN, weight_C5,&status);//120@5*5 × 16(S4) 
	clEnqueueWriteBuffer(cmdQueue, clbuf_weight_C5, CL_FALSE,0,sizeof(float)*len_weight_C5_CNN,weight_C5,0,NULL,NULL);
    clbuf_bias_C5 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_bias_C5_CNN, bias_C5, &status);//120  
	clEnqueueWriteBuffer(cmdQueue, clbuf_bias_C5, CL_FALSE,0,sizeof(float)*len_bias_C5_CNN,bias_C5,0,NULL,NULL);
	clbuf_neuron_C5 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*num_neuron_C5_CNN, NULL, &status);//120@1*1
    //Forward_output()
	clbuf_weight_output = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_weight_output_CNN, weight_output, &status);//120×10
	clEnqueueWriteBuffer(cmdQueue, clbuf_weight_output, CL_FALSE,0,sizeof(float)*len_weight_output_CNN,weight_output,0,NULL,NULL);
	clbuf_bias_output = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_bias_output_CNN, bias_output, &status);//10
	clEnqueueWriteBuffer(cmdQueue, clbuf_bias_output, CL_FALSE,0,sizeof(float)*len_bias_output_CNN,bias_output,0,NULL,NULL);
    clbuf_neuron_output = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*num_neuron_output_CNN, NULL, &status);//10
    //Backward_output()
    clbuf_delta_neuron_output = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*num_neuron_output_CNN, NULL, &status);//10
    clbuf_data_single_label = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*num_neuron_output_CNN, NULL, &status);//10
    //Backward_C5()
    clbuf_delta_neuron_C5 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*num_neuron_C5_CNN, NULL, &status);//120
    clbuf_delta_weight_output = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_weight_output_CNN, NULL, &status);//1200
    //Backward_S4()
    clbuf_delta_neuron_S4 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*num_neuron_S4_CNN, NULL, &status);//400
    clbuf_delta_weight_C5 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_weight_C5_CNN, NULL, &status);//48000
    //Backward_C3()
    clbuf_delta_neuron_C3 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*num_neuron_C3_CNN, NULL, &status);//1600
    clbuf_delta_weight_S4 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_weight_S4_CNN, NULL, &status);//16
    //Backward_S2()
    clbuf_delta_neuron_S2 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*num_neuron_S2_CNN, NULL, &status);//1176
    clbuf_delta_weight_C3 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_weight_C3_CNN, NULL, &status);//2400
    //Backward_C1()
    clbuf_delta_neuron_C1 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*num_neuron_C1_CNN, NULL, &status);//4704
    clbuf_delta_weight_S2 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_weight_S2_CNN, NULL, &status);//6
    //Backward_input()
    clbuf_delta_neuron_input = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*num_neuron_input_CNN, NULL, &status);//1024
    clbuf_delta_weight_C1 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_weight_C1_CNN, NULL, &status);//150
	//DeltaBias()
    clbuf_delta_bias_C1 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_bias_C1_CNN, NULL, &status);
	clbuf_delta_bias_S2 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_bias_S2_CNN, NULL, &status);
	clbuf_delta_bias_C3 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_bias_C3_CNN, NULL, &status);
	clbuf_delta_bias_S4 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_bias_S4_CNN, NULL, &status);
	clbuf_delta_bias_C5 = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_bias_C5_CNN, NULL, &status);
	clbuf_delta_bias_output = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float)*len_bias_output_CNN, NULL, &status);
	//UpdateWeights()
	clbuf_E_weight_output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * len_weight_output_CNN, E_weight_output, &status);
	clbuf_E_weight_C5 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * len_weight_C5_CNN, E_weight_C5, &status);
	clbuf_E_weight_S4 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * len_weight_S4_CNN, E_weight_S4, &status);
	clbuf_E_weight_C3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * len_weight_C3_CNN, E_weight_C3, &status);
	clbuf_E_weight_S2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * len_weight_S2_CNN, E_weight_S2, &status);
	clbuf_E_weight_C1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * len_weight_C1_CNN, E_weight_C1, &status);

	clbuf_E_bias_output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * len_bias_output_CNN, E_bias_output, &status);
	clbuf_E_bias_C5 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * len_bias_C5_CNN, E_bias_C5, &status);
	clbuf_E_bias_S4 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * len_bias_S4_CNN, E_bias_S4, &status);
	clbuf_E_bias_C3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * len_bias_C3_CNN, E_bias_C3, &status);
	clbuf_E_bias_S2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * len_bias_S2_CNN, E_bias_S2, &status);
	clbuf_E_bias_C1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * len_bias_C1_CNN, E_bias_C1, &status);
	//-----------------------------------------------------
    // STEP 7: Create and compile the program
    //----------------------------------------------------- 
    //kernel.cl
    const char * filename1 = "Kernel/kernel.cl";
    std::string sourceStr;
    status = convertToString(filename1, sourceStr);
    const char * source1 = sourceStr.c_str();
    size_t sourceSize[] = { strlen(source1) }; 
    // Create a program using clCreateProgramWithSource()
    program1 = clCreateProgramWithSource(context,1,&source1,sourceSize,NULL);
    status = clBuildProgram(program1,numDevices,devices,NULL,NULL,NULL);
    if(status != 0){
        printf("\nclBuild1 failed:%d\n", status);
        char tbuf[0x10000];
        clGetProgramBuildInfo(program1, devices[0], CL_PROGRAM_BUILD_LOG, 0x10000, tbuf, NULL);
        printf("\n%s\n", tbuf);
		exit(1);
        return false;
    }
    //backward.cl
    const char * filename2 = "Kernel/backward.cl";
    status = convertToString(filename2, sourceStr);
    const char * source2 = sourceStr.c_str();
    sourceSize[0] = { strlen(source2) }; 
    // Create a program using clCreateProgramWithSource()
    program2 = clCreateProgramWithSource(context,1,&source2,sourceSize,NULL);
    status = clBuildProgram(program2,numDevices,devices,NULL,NULL,NULL);
    if(status != 0){
        printf("\nclBuild2 failed:%d\n", status);
        char tbuf[0x10000];
        clGetProgramBuildInfo(program2, devices[0], CL_PROGRAM_BUILD_LOG, 0x10000, tbuf, NULL);
        printf("\n%s\n", tbuf);
		exit(1);
        return false;
    }
    //update_delta.cl
    const char * filename3 = "Kernel/update_delta.cl";
    status = convertToString(filename3, sourceStr);
    const char * source3 = sourceStr.c_str();
    sourceSize[0] = { strlen(source3) }; 
    // Create a program using clCreateProgramWithSource()
    program3 = clCreateProgramWithSource(context,1,&source3,sourceSize,NULL);
    status = clBuildProgram(program3,numDevices,devices,NULL,NULL,NULL);
    if(status != 0){
        printf("\nclBuild3 failed:%d\n", status);
        char tbuf[0x10000];
        clGetProgramBuildInfo(program3, devices[0], CL_PROGRAM_BUILD_LOG, 0x10000, tbuf, NULL);
        printf("\n%s\n", tbuf);
		exit(1);
        return false;
    }
    //-----------------------------------------------------
    // STEP 8: Create the kernel
    //----------------------------------------------------- 
    kernel1  = clCreateKernel(program1, "kernel_forward_c1", &status);
    kernel2  = clCreateKernel(program1, "kernel_forward_s2", &status);
    kernel3  = clCreateKernel(program1, "kernel_forward_c3", &status);
    kernel4  = clCreateKernel(program1, "kernel_forward_s4", &status);
    kernel5  = clCreateKernel(program1, "kernel_forward_c5", &status);
    kernel6  = clCreateKernel(program1, "kernel_forward_output", &status);
	kernel7  = clCreateKernel(program2, "back_output", &status);
	kernel8  = clCreateKernel(program2, "back_c5", &status);
	kernel9  = clCreateKernel(program2, "back_s4", &status);
	kernel10 = clCreateKernel(program2, "back_c3", &status);
	kernel11  = clCreateKernel(program2, "back1_s2", &status);
	kernel12  = clCreateKernel(program2, "back2_s2", &status);
	kernel13  = clCreateKernel(program2, "back_c1", &status);
	kernel14  = clCreateKernel(program2, "back1_input", &status);
	kernel15  = clCreateKernel(program2, "back2_input", &status);
	kernel16 = clCreateKernel(program3, "deltabias", &status);
	kernel17 = clCreateKernel(program3, "update_wb", &status);
    //-----------------------------------------------------
    // STEP 9: Set the kernel arguments
    //----------------------------------------------------- 
    //Forward_C1()
    int cl_num_map_C1_CNN = num_map_C1_CNN;
    int cl_width_image_C1_CNN = width_image_C1_CNN;
    int cl_height_image_C1_CNN = height_image_C1_CNN;
    int cl_width_kernel_conv_CNN = width_kernel_conv_CNN;
    int cl_height_kernel_conv_CNN = height_kernel_conv_CNN;	
    int cl_num_map_input_CNN = num_map_input_CNN;
    int cl_width_image_input_CNN = width_image_input_CNN;
    int cl_height_image_input_CNN = height_image_input_CNN;
    status  = clSetKernelArg(kernel1, 0, sizeof(cl_mem), &clbuf_data_single_image);
	status |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &clbuf_weight_C1);
	status |= clSetKernelArg(kernel1, 2, sizeof(cl_mem), &clbuf_bias_C1);
	status |= clSetKernelArg(kernel1, 3, sizeof(cl_mem), &clbuf_neuron_C1);
    status |= clSetKernelArg(kernel1, 4, sizeof(int), &cl_num_map_C1_CNN);
    status |= clSetKernelArg(kernel1, 5, sizeof(int), &cl_width_image_C1_CNN);
    status |= clSetKernelArg(kernel1, 6, sizeof(int), &cl_height_image_C1_CNN);
	status |= clSetKernelArg(kernel1, 7, sizeof(int), &cl_width_kernel_conv_CNN);
	status |= clSetKernelArg(kernel1, 8, sizeof(int), &cl_height_kernel_conv_CNN);
	status |= clSetKernelArg(kernel1, 9, sizeof(int), &cl_num_map_input_CNN);
	status |= clSetKernelArg(kernel1, 10, sizeof(int), &cl_width_image_input_CNN);
	status |= clSetKernelArg(kernel1, 11, sizeof(int), &cl_height_image_input_CNN);
    if(status) printf("set kernel1 parameter error\n");    
    //Forward_S2()
    int cl_num_map_S2_CNN = num_map_S2_CNN;
    int cl_width_image_S2_CNN = width_image_S2_CNN;
    int cl_height_image_S2_CNN = height_image_S2_CNN;
    int cl_width_kernel_pooling_CNN = width_kernel_pooling_CNN;
    int cl_height_kernel_pooling_CNN = height_kernel_pooling_CNN;	
    cl_num_map_C1_CNN = num_map_C1_CNN;
    cl_width_image_C1_CNN = width_image_C1_CNN;
    cl_height_image_C1_CNN = height_image_C1_CNN;
    status  = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &clbuf_neuron_C1);
	status |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &clbuf_weight_S2);
	status |= clSetKernelArg(kernel2, 2, sizeof(cl_mem), &clbuf_bias_S2);
	status |= clSetKernelArg(kernel2, 3, sizeof(cl_mem), &clbuf_neuron_S2);
    status |= clSetKernelArg(kernel2, 4, sizeof(int), &cl_num_map_S2_CNN);
    status |= clSetKernelArg(kernel2, 5, sizeof(int), &cl_width_image_S2_CNN);
    status |= clSetKernelArg(kernel2, 6, sizeof(int), &cl_height_image_S2_CNN);
	status |= clSetKernelArg(kernel2, 7, sizeof(int), &cl_width_kernel_pooling_CNN);
	status |= clSetKernelArg(kernel2, 8, sizeof(int), &cl_height_kernel_pooling_CNN);
	status |= clSetKernelArg(kernel2, 9, sizeof(int), &cl_num_map_C1_CNN);
	status |= clSetKernelArg(kernel2, 10, sizeof(int), &cl_width_image_C1_CNN);
	status |= clSetKernelArg(kernel2, 11, sizeof(int), &cl_height_image_C1_CNN);
    if(status) printf("set kernel2 parameter error\n");
    //Forward_C3()
    int cl_num_map_C3_CNN = num_map_C3_CNN;
    int cl_width_image_C3_CNN = width_image_C3_CNN;
    int cl_height_image_C3_CNN = height_image_C3_CNN;
    cl_width_kernel_conv_CNN = width_kernel_conv_CNN;
    cl_height_kernel_conv_CNN = height_kernel_conv_CNN;	
    cl_num_map_S2_CNN = num_map_S2_CNN;
    cl_width_image_S2_CNN = width_image_S2_CNN;
    cl_height_image_S2_CNN = height_image_S2_CNN;
    status  = clSetKernelArg(kernel3, 0, sizeof(cl_mem),&clbuf_neuron_S2);
	status |= clSetKernelArg(kernel3, 1, sizeof(cl_mem), &clbuf_weight_C3);
	status |= clSetKernelArg(kernel3, 2, sizeof(cl_mem), &clbuf_bias_C3);
	status |= clSetKernelArg(kernel3, 3, sizeof(cl_mem), &clbuf_neuron_C3);
    status |= clSetKernelArg(kernel3, 4, sizeof(int), &cl_num_map_C3_CNN);
    status |= clSetKernelArg(kernel3, 5, sizeof(int), &cl_width_image_C3_CNN);
    status |= clSetKernelArg(kernel3, 6, sizeof(int), &cl_height_image_C3_CNN);
	status |= clSetKernelArg(kernel3, 7, sizeof(int), &cl_width_kernel_conv_CNN);
	status |= clSetKernelArg(kernel3, 8, sizeof(int), &cl_height_kernel_conv_CNN);
	status |= clSetKernelArg(kernel3, 9, sizeof(int), &cl_num_map_S2_CNN);
	status |= clSetKernelArg(kernel3, 10, sizeof(int), &cl_width_image_S2_CNN);
	status |= clSetKernelArg(kernel3, 11, sizeof(int), &cl_height_image_S2_CNN);
	status |= clSetKernelArg(kernel3, 12, sizeof(cl_mem), &clbuf_tbl);
    if(status) printf("set kernel3 parameter error\n");
    //Forward_S4()
    int cl_num_map_S4_CNN = num_map_S4_CNN;
    int cl_width_image_S4_CNN = width_image_S4_CNN;
    int cl_height_image_S4_CNN = height_image_S4_CNN;
    cl_width_kernel_pooling_CNN = width_kernel_pooling_CNN;
    cl_height_kernel_pooling_CNN = height_kernel_pooling_CNN;	
    cl_num_map_C3_CNN = num_map_C3_CNN;
    cl_width_image_C3_CNN = width_image_C3_CNN;
    cl_height_image_C3_CNN = height_image_C3_CNN;
    status  = clSetKernelArg(kernel4, 0, sizeof(cl_mem),&clbuf_neuron_C3);
	status |= clSetKernelArg(kernel4, 1, sizeof(cl_mem), &clbuf_weight_S4);
	status |= clSetKernelArg(kernel4, 2, sizeof(cl_mem), &clbuf_bias_S4);
	status |= clSetKernelArg(kernel4, 3, sizeof(cl_mem), &clbuf_neuron_S4);
    status |= clSetKernelArg(kernel4, 4, sizeof(int), &cl_num_map_S4_CNN);
    status |= clSetKernelArg(kernel4, 5, sizeof(int), &cl_width_image_S4_CNN);
    status |= clSetKernelArg(kernel4, 6, sizeof(int), &cl_height_image_S4_CNN);
	status |= clSetKernelArg(kernel4, 7, sizeof(int), &cl_width_kernel_pooling_CNN);
	status |= clSetKernelArg(kernel4, 8, sizeof(int), &cl_height_kernel_pooling_CNN);
	status |= clSetKernelArg(kernel4, 9, sizeof(int), &cl_num_map_C3_CNN);
	status |= clSetKernelArg(kernel4, 10, sizeof(int), &cl_width_image_C3_CNN);
	status |= clSetKernelArg(kernel4, 11, sizeof(int), &cl_height_image_C3_CNN);
    if(status) printf("set kernel4 parameter error\n");
    //Forward_C5()
    int cl_num_map_C5_CNN = num_map_C5_CNN;
    int cl_width_image_C5_CNN = width_image_C5_CNN;
    int cl_height_image_C5_CNN = height_image_C5_CNN;
    cl_width_kernel_conv_CNN = width_kernel_conv_CNN;
    cl_height_kernel_conv_CNN = height_kernel_conv_CNN;	
    cl_num_map_S4_CNN = num_map_S4_CNN;
    cl_width_image_S4_CNN = width_image_S4_CNN;
    cl_height_image_S4_CNN = height_image_S4_CNN;
    status  = clSetKernelArg(kernel5, 0, sizeof(cl_mem), &clbuf_neuron_S4);
	status |= clSetKernelArg(kernel5, 1, sizeof(cl_mem), &clbuf_weight_C5);
	status |= clSetKernelArg(kernel5, 2, sizeof(cl_mem), &clbuf_bias_C5);
	status |= clSetKernelArg(kernel5, 3, sizeof(cl_mem), &clbuf_neuron_C5);
    status |= clSetKernelArg(kernel5, 4, sizeof(int), &cl_num_map_C5_CNN);
    status |= clSetKernelArg(kernel5, 5, sizeof(int), &cl_width_image_C5_CNN);
    status |= clSetKernelArg(kernel5, 6, sizeof(int), &cl_height_image_C5_CNN);
	status |= clSetKernelArg(kernel5, 7, sizeof(int), &cl_width_kernel_conv_CNN);
	status |= clSetKernelArg(kernel5, 8, sizeof(int), &cl_height_kernel_conv_CNN);
	status |= clSetKernelArg(kernel5, 9, sizeof(int), &cl_num_map_S4_CNN);
	status |= clSetKernelArg(kernel5, 10, sizeof(int), &cl_width_image_S4_CNN);
	status |= clSetKernelArg(kernel5, 11, sizeof(int), &cl_height_image_S4_CNN);
    if(status) printf("set kernel5 parameter error\n");
    //Forward_output()
    int cl_num_map_output_CNN = num_map_output_CNN;	
    cl_num_map_C5_CNN = num_map_C5_CNN;
    status  = clSetKernelArg(kernel6, 0, sizeof(cl_mem), &clbuf_neuron_C5);
	status |= clSetKernelArg(kernel6, 1, sizeof(cl_mem), &clbuf_weight_output);
	status |= clSetKernelArg(kernel6, 2, sizeof(cl_mem), &clbuf_bias_output);
	status |= clSetKernelArg(kernel6, 3, sizeof(cl_mem), &clbuf_neuron_output);
    status |= clSetKernelArg(kernel6, 4, sizeof(int), &cl_num_map_output_CNN);
	status |= clSetKernelArg(kernel6, 5, sizeof(int), &cl_num_map_C5_CNN);
    if(status) printf("set kernel6 parameter error\n");
    //Backward_output()
    status  = clSetKernelArg(kernel7, 0, sizeof(cl_mem), &clbuf_neuron_output);
	status |= clSetKernelArg(kernel7, 1, sizeof(cl_mem), &clbuf_data_single_label);
	status |= clSetKernelArg(kernel7, 2, sizeof(cl_mem), &clbuf_delta_bias_output);
    if(status) printf("set kernel7 parameter error\n");
    //Backward_C5()
    status  = clSetKernelArg(kernel8, 0, sizeof(cl_mem), &clbuf_delta_bias_output);
	status |= clSetKernelArg(kernel8, 1, sizeof(cl_mem), &clbuf_weight_output);
	status |= clSetKernelArg(kernel8, 2, sizeof(cl_mem), &clbuf_neuron_C5);
	status |= clSetKernelArg(kernel8, 3, sizeof(cl_mem), &clbuf_delta_bias_C5);
	status |= clSetKernelArg(kernel8, 4, sizeof(cl_mem), &clbuf_delta_weight_output);
    if(status) printf("set kernel8 parameter error\n"); 
    //Backward_S4()
    status  = clSetKernelArg(kernel9, 0, sizeof(cl_mem), &clbuf_weight_C5);
	status |= clSetKernelArg(kernel9, 1, sizeof(cl_mem), &clbuf_delta_bias_C5);
	status |= clSetKernelArg(kernel9, 2, sizeof(cl_mem), &clbuf_neuron_S4);
	status |= clSetKernelArg(kernel9, 3, sizeof(cl_mem), &clbuf_delta_weight_C5);
	status |= clSetKernelArg(kernel9, 4, sizeof(cl_mem), &clbuf_delta_neuron_S4);
    if(status) printf("set kernel9 parameter error\n");
    //Backward_C3()
    status  = clSetKernelArg(kernel10, 0, sizeof(cl_mem), &clbuf_weight_S4);
	status |= clSetKernelArg(kernel10, 1, sizeof(cl_mem), &clbuf_delta_neuron_S4);
	status |= clSetKernelArg(kernel10, 2, sizeof(cl_mem), &clbuf_neuron_C3);
	status |= clSetKernelArg(kernel10, 3, sizeof(cl_mem), &clbuf_delta_neuron_C3);
	status |= clSetKernelArg(kernel10, 4, sizeof(cl_mem), &clbuf_delta_weight_S4);
    if(status) printf("set kernel10 parameter error\n");      
    //Backward_S2()
    status  = clSetKernelArg(kernel11, 0, sizeof(cl_mem), &clbuf_weight_C3);
	status |= clSetKernelArg(kernel11, 1, sizeof(cl_mem), &clbuf_delta_neuron_C3);
	status |= clSetKernelArg(kernel11, 2, sizeof(cl_mem), &clbuf_neuron_S2);
	status |= clSetKernelArg(kernel11, 3, sizeof(cl_mem), &clbuf_delta_neuron_S2);
    if(status) printf("set kernel11 parameter error\n");
    status  = clSetKernelArg(kernel12, 0, sizeof(cl_mem), &clbuf_neuron_S2);
	status |= clSetKernelArg(kernel12, 1, sizeof(cl_mem), &clbuf_delta_neuron_C3);
	status |= clSetKernelArg(kernel12, 2, sizeof(cl_mem), &clbuf_delta_weight_C3);
    if(status) printf("set kernel12 parameter error\n"); 
    //Backward_C1()
    status  = clSetKernelArg(kernel13, 0, sizeof(cl_mem), &clbuf_delta_neuron_S2);
	status |= clSetKernelArg(kernel13, 1, sizeof(cl_mem), &clbuf_neuron_C1);
	status |= clSetKernelArg(kernel13, 2, sizeof(cl_mem), &clbuf_weight_S2);
	status |= clSetKernelArg(kernel13, 3, sizeof(cl_mem), &clbuf_delta_neuron_C1);
	status |= clSetKernelArg(kernel13, 4, sizeof(cl_mem), &clbuf_delta_weight_S2);
    if(status) printf("set kernel13 parameter error\n");
    //Backward_Input()   
    status  = clSetKernelArg(kernel14, 0, sizeof(cl_mem), &clbuf_weight_C1);
	status |= clSetKernelArg(kernel14, 1, sizeof(cl_mem), &clbuf_delta_neuron_C1);
	status |= clSetKernelArg(kernel14, 2, sizeof(cl_mem), &clbuf_data_single_image);
	status |= clSetKernelArg(kernel14, 3, sizeof(cl_mem), &clbuf_delta_neuron_input);
    if(status) printf("set kernel14 parameter error\n");
    status = clSetKernelArg(kernel15, 0, sizeof(cl_mem), &clbuf_data_single_image);
	status |= clSetKernelArg(kernel15, 1, sizeof(cl_mem), &clbuf_delta_neuron_C1);
	status |= clSetKernelArg(kernel15, 2, sizeof(cl_mem), &clbuf_delta_weight_C1);
    if(status) printf("set kernel15 parameter error\n");
    //DeltaBias()
    status = clSetKernelArg(kernel16, 0, sizeof(cl_mem), &clbuf_delta_neuron_S4);
	status |= clSetKernelArg(kernel16, 1, sizeof(cl_mem), &clbuf_delta_neuron_C3);
	status |= clSetKernelArg(kernel16, 2, sizeof(cl_mem), &clbuf_delta_neuron_S2);
	status |= clSetKernelArg(kernel16, 3, sizeof(cl_mem), &clbuf_delta_neuron_C1);
	status |= clSetKernelArg(kernel16, 4, sizeof(cl_mem), &clbuf_delta_bias_S4);
	status |= clSetKernelArg(kernel16, 5, sizeof(cl_mem), &clbuf_delta_bias_C3);
	status |= clSetKernelArg(kernel16, 6, sizeof(cl_mem), &clbuf_delta_bias_S2);
	status |= clSetKernelArg(kernel16, 7, sizeof(cl_mem), &clbuf_delta_bias_C1);
    if(status) printf("set kernel16 parameter error\n"); 
    //UpdateWeights()
    status = clSetKernelArg(kernel17, 0, sizeof(cl_mem), &clbuf_delta_weight_C1);
	status |= clSetKernelArg(kernel17, 1, sizeof(cl_mem), &clbuf_delta_bias_C1);
	status |= clSetKernelArg(kernel17, 2, sizeof(cl_mem), &clbuf_delta_weight_S2);
	status |= clSetKernelArg(kernel17, 3, sizeof(cl_mem), &clbuf_delta_bias_S2);
	status |= clSetKernelArg(kernel17, 4, sizeof(cl_mem), &clbuf_delta_weight_C3);
	status |= clSetKernelArg(kernel17, 5, sizeof(cl_mem), &clbuf_delta_bias_C3);
	status |= clSetKernelArg(kernel17, 6, sizeof(cl_mem), &clbuf_delta_weight_S4);
	status |= clSetKernelArg(kernel17, 7, sizeof(cl_mem), &clbuf_delta_bias_S4);
	status |= clSetKernelArg(kernel17, 8, sizeof(cl_mem), &clbuf_delta_weight_C5);
	status |= clSetKernelArg(kernel17, 9, sizeof(cl_mem), &clbuf_delta_bias_C5);
	status |= clSetKernelArg(kernel17, 10, sizeof(cl_mem), &clbuf_delta_weight_output);
	status |= clSetKernelArg(kernel17, 11, sizeof(cl_mem), &clbuf_delta_bias_output);
	
	status |= clSetKernelArg(kernel17, 12, sizeof(cl_mem), &clbuf_E_weight_C1);
	status |= clSetKernelArg(kernel17, 13, sizeof(cl_mem), &clbuf_E_bias_C1);
	status |= clSetKernelArg(kernel17, 14, sizeof(cl_mem), &clbuf_E_weight_S2);
	status |= clSetKernelArg(kernel17, 15, sizeof(cl_mem), &clbuf_E_bias_S2);
	status |= clSetKernelArg(kernel17, 16, sizeof(cl_mem), &clbuf_E_weight_C3);
	status |= clSetKernelArg(kernel17, 17, sizeof(cl_mem), &clbuf_E_bias_C3);
	status |= clSetKernelArg(kernel17, 18, sizeof(cl_mem), &clbuf_E_weight_S4);
	status |= clSetKernelArg(kernel17, 19, sizeof(cl_mem), &clbuf_E_bias_S4);
	status |= clSetKernelArg(kernel17, 20, sizeof(cl_mem), &clbuf_E_weight_C5);
	status |= clSetKernelArg(kernel17, 21, sizeof(cl_mem), &clbuf_E_bias_C5);
	status |= clSetKernelArg(kernel17, 22, sizeof(cl_mem), &clbuf_E_weight_output);
	status |= clSetKernelArg(kernel17, 23, sizeof(cl_mem), &clbuf_E_bias_output);

	status |= clSetKernelArg(kernel17, 24, sizeof(cl_mem), &clbuf_weight_C1);
	status |= clSetKernelArg(kernel17, 25, sizeof(cl_mem), &clbuf_bias_C1);
	status |= clSetKernelArg(kernel17, 26, sizeof(cl_mem), &clbuf_weight_S2);
	status |= clSetKernelArg(kernel17, 27, sizeof(cl_mem), &clbuf_bias_S2);
	status |= clSetKernelArg(kernel17, 28, sizeof(cl_mem), &clbuf_weight_C3);
	status |= clSetKernelArg(kernel17, 29, sizeof(cl_mem), &clbuf_bias_C3);
	status |= clSetKernelArg(kernel17, 30, sizeof(cl_mem), &clbuf_weight_S4);
	status |= clSetKernelArg(kernel17, 31, sizeof(cl_mem), &clbuf_bias_S4);
	status |= clSetKernelArg(kernel17, 32, sizeof(cl_mem), &clbuf_weight_C5);
	status |= clSetKernelArg(kernel17, 33, sizeof(cl_mem), &clbuf_bias_C5);
	status |= clSetKernelArg(kernel17, 34, sizeof(cl_mem), &clbuf_weight_output);
	status |= clSetKernelArg(kernel17, 35, sizeof(cl_mem), &clbuf_bias_output);
    if(status) printf("set kernel17 parameter error\n");

	return true;
}
void OpenCL_free(){
    //-----------------------------------------------------
    // STEP 13: Release OpenCL resources
    //----------------------------------------------------- 
    
    // Free OpenCL resources
    clReleaseProgram(program1);
    clReleaseProgram(program2);
    clReleaseProgram(program3);
    clReleaseCommandQueue(cmdQueue);
	clReleaseContext(context);
	//kernel
	///*
	clReleaseKernel(kernel1);
	clReleaseKernel(kernel2);
	clReleaseKernel(kernel3);
	clReleaseKernel(kernel4);
    clReleaseKernel(kernel5);
    clReleaseKernel(kernel6);
    clReleaseKernel(kernel7);
    clReleaseKernel(kernel8);
    clReleaseKernel(kernel9);
    clReleaseKernel(kernel10);
    clReleaseKernel(kernel11);
    clReleaseKernel(kernel12);
	clReleaseKernel(kernel13);
	clReleaseKernel(kernel14);
	clReleaseKernel(kernel15);
	clReleaseKernel(kernel16);
	clReleaseKernel(kernel17);
	//*/
	//Forward_C1()
    clReleaseMemObject(clbuf_data_single_image);
    clReleaseMemObject(clbuf_weight_C1);
    clReleaseMemObject(clbuf_bias_C1);
	clReleaseMemObject(clbuf_neuron_C1);
	//Forward_S2()
	clReleaseMemObject(clbuf_weight_S2);
    clReleaseMemObject(clbuf_bias_S2);
	clReleaseMemObject(clbuf_neuron_S2);
	//Forward_C3()
    clReleaseMemObject(clbuf_weight_C3);
    clReleaseMemObject(clbuf_bias_C3);
	clReleaseMemObject(clbuf_neuron_C3);
	clReleaseMemObject(clbuf_tbl);
	//Forward_S4()
	clReleaseMemObject(clbuf_weight_S4);
    clReleaseMemObject(clbuf_bias_S4);
	clReleaseMemObject(clbuf_neuron_S4);
	//Forward_C5()
    clReleaseMemObject(clbuf_weight_C5);
    clReleaseMemObject(clbuf_bias_C5);
	clReleaseMemObject(clbuf_neuron_C5);
    //Forward_output()
    clReleaseMemObject(clbuf_weight_output);
    clReleaseMemObject(clbuf_bias_output);
	clReleaseMemObject(clbuf_neuron_output);
    //Backward_output()
    clReleaseMemObject(clbuf_delta_neuron_output);
    clReleaseMemObject(clbuf_data_single_label);
    //Backward_C5()
    clReleaseMemObject(clbuf_delta_neuron_C5);
    clReleaseMemObject(clbuf_delta_weight_output);
    //Backward_S4()
    clReleaseMemObject(clbuf_delta_neuron_S4);
    clReleaseMemObject(clbuf_delta_weight_C5);
    //clReleaseMemObject(clbuf_delta_bias_C5);
    //Backward_C3()
    clReleaseMemObject(clbuf_delta_neuron_C3);
    clReleaseMemObject(clbuf_delta_weight_S4);
    //Backward_S2()
    clReleaseMemObject(clbuf_delta_neuron_S2);
    clReleaseMemObject(clbuf_delta_weight_C3);
    //Backward_C1()
    clReleaseMemObject(clbuf_delta_neuron_C1);
    clReleaseMemObject(clbuf_delta_weight_S2);
    //Backward_Input()
    clReleaseMemObject(clbuf_delta_neuron_input);
    clReleaseMemObject(clbuf_delta_weight_C1);
    //DeltaBias()
	clReleaseMemObject(clbuf_delta_bias_output);
	clReleaseMemObject(clbuf_delta_bias_C5);
	clReleaseMemObject(clbuf_delta_bias_S4);
	clReleaseMemObject(clbuf_delta_bias_C3);
	clReleaseMemObject(clbuf_delta_bias_S2);
	clReleaseMemObject(clbuf_delta_bias_C1);
    //UpdateWeights()
    clReleaseMemObject(clbuf_E_weight_output);
    clReleaseMemObject(clbuf_E_weight_C5);
    clReleaseMemObject(clbuf_E_weight_S4);
    clReleaseMemObject(clbuf_E_weight_C3);
    clReleaseMemObject(clbuf_E_weight_S2);
    clReleaseMemObject(clbuf_E_weight_C1);
	
    clReleaseMemObject(clbuf_E_bias_output);
    clReleaseMemObject(clbuf_E_bias_C5);
    clReleaseMemObject(clbuf_E_bias_S4);
    clReleaseMemObject(clbuf_E_bias_C3);
    clReleaseMemObject(clbuf_E_bias_S2);
    clReleaseMemObject(clbuf_E_bias_C1);

    // Free host resources
    free(platforms);
    free(devices);
}

int main(int argc,char *argv[]){
	CNN Tcnn;//实例化一个对象
	Tcnn.init();//初始化
	//OpenCL_init();
	Tcnn.train();//训练
	OpenCL_free();
	return 0;
}
