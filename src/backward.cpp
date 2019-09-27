/*
 * backward.cpp
 *
 *  Created on: Apr 29, 2017
 *      Author: copper
 */
#include "cnn.h"

using namespace std;

// connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
static const bool tbl[6][16] = {
	O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
	O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
	O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
	X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
	X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
	X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
};
#undef O
#undef X

void CNN::check_error(cl_int error) {
    if (error != CL_SUCCESS) {
		switch (error) {
		case CL_DEVICE_NOT_FOUND: printf("Error: Device not found.\n"); break;
		case CL_DEVICE_NOT_AVAILABLE: printf("Error: Device not available\n"); break;
		case CL_COMPILER_NOT_AVAILABLE: printf("Error: Compiler not available\n"); break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE: printf("Error: Memory object allocation failure\n"); break;
		case CL_OUT_OF_RESOURCES: printf("Error: Out of resources\n"); break;
		case CL_OUT_OF_HOST_MEMORY: printf("Error: Out of host memory\n"); break;
		case CL_PROFILING_INFO_NOT_AVAILABLE: printf("Error: Profiling information not available\n"); break;
		case CL_MEM_COPY_OVERLAP: printf("Error: Memory copy overlap\n"); break;
		case CL_IMAGE_FORMAT_MISMATCH: printf("Error: Image format mismatch\n"); break;
		case CL_IMAGE_FORMAT_NOT_SUPPORTED: printf("Error: Image format not supported\n"); break;
		case CL_BUILD_PROGRAM_FAILURE: printf("Error: Program build failure\n"); break;
		case CL_MAP_FAILURE: printf("Error: Map failure\n"); break;
		case CL_INVALID_VALUE: printf("Error: Invalid value\n"); break;
		case CL_INVALID_DEVICE_TYPE: printf("Error: Invalid device type\n"); break;
		case CL_INVALID_PLATFORM: printf("Error: Invalid platform\n"); break;
		case CL_INVALID_DEVICE: printf("Error: Invalid device\n"); break;
		case CL_INVALID_CONTEXT: printf("Error: Invalid context\n"); break;
		case CL_INVALID_QUEUE_PROPERTIES: printf("Error: Invalid queue properties\n"); break;
		case CL_INVALID_COMMAND_QUEUE: printf("Error: Invalid command queue\n"); break;
		case CL_INVALID_HOST_PTR: printf("Error: Invalid host pointer\n"); break;
		case CL_INVALID_MEM_OBJECT: printf("Error: Invalid memory object\n"); break;
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: printf("Error: Invalid image format descriptor\n"); break;
		case CL_INVALID_IMAGE_SIZE: printf("Error: Invalid image size\n"); break;
		case CL_INVALID_SAMPLER: printf("Error: Invalid sampler\n"); break;
		case CL_INVALID_BINARY: printf("Error: Invalid binary\n"); break;
		case CL_INVALID_BUILD_OPTIONS: printf("Error: Invalid build options\n"); break;
		case CL_INVALID_PROGRAM: printf("Error: Invalid program\n"); break;
		case CL_INVALID_PROGRAM_EXECUTABLE: printf("Error: Invalid program executable\n"); break;
		case CL_INVALID_KERNEL_NAME: printf("Error: Invalid kernel name\n"); break;
		case CL_INVALID_KERNEL_DEFINITION: printf("Error: Invalid kernel definition\n"); break;
		case CL_INVALID_KERNEL: printf("Error: Invalid kernel\n"); break;
		case CL_INVALID_ARG_INDEX: printf("Error: Invalid argument index\n"); break;
		case CL_INVALID_ARG_VALUE: printf("Error: Invalid argument value\n"); break;
		case CL_INVALID_ARG_SIZE: printf("Error: Invalid argument size\n"); break;
		case CL_INVALID_KERNEL_ARGS: printf("Error: Invalid kernel arguments\n"); break;
		case CL_INVALID_WORK_DIMENSION: printf("Error: Invalid work dimensionsension\n"); break;
		case CL_INVALID_WORK_GROUP_SIZE: printf("Error: Invalid work group size\n"); break;
		case CL_INVALID_WORK_ITEM_SIZE: printf("Error: Invalid work item size\n"); break;
		case CL_INVALID_GLOBAL_OFFSET: printf("Error: Invalid global offset\n"); break;
		case CL_INVALID_EVENT_WAIT_LIST: printf("Error: Invalid event wait list\n"); break;
		case CL_INVALID_EVENT: printf("Error: Invalid event\n"); break;
		case CL_INVALID_OPERATION: printf("Error: Invalid operation\n"); break;
		case CL_INVALID_GL_OBJECT: printf("Error: Invalid OpenGL object\n"); break;
		case CL_INVALID_BUFFER_SIZE: printf("Error: Invalid buffer size\n"); break;
		case CL_INVALID_MIP_LEVEL: printf("Error: Invalid mip-map level\n"); break;
		case -1024: printf("Error: *clBLAS* Functionality is not implemented\n"); break;
		case -1023: printf("Error: *clBLAS* Library is not initialized yet\n"); break;
		case -1022: printf("Error: *clBLAS* Matrix A is not a valid memory object\n"); break;
		case -1021: printf("Error: *clBLAS* Matrix B is not a valid memory object\n"); break;
		case -1020: printf("Error: *clBLAS* Matrix C is not a valid memory object\n"); break;
		case -1019: printf("Error: *clBLAS* Vector X is not a valid memory object\n"); break;
		case -1018: printf("Error: *clBLAS* Vector Y is not a valid memory object\n");break;
		case -1017: printf("Error: *clBLAS* An input dimension (M,N,K) is invalid\n"); break;
		case -1016: printf("Error: *clBLAS* Leading dimension A must not be less than the size of the first dimension\n"); break;
		case -1015: printf("Error: *clBLAS* Leading dimension B must not be less than the size of the second dimension\n"); break;
		case -1014: printf("Error: *clBLAS* Leading dimension C must not be less than the size of the third dimension\n"); break;
		case -1013: printf("Error: *clBLAS* The increment for a vector X must not be 0\n"); break;
		case -1012: printf("Error: *clBLAS* The increment for a vector Y must not be 0\n"); break;
		case -1011: printf("Error: *clBLAS* The memory object for Matrix A is too small\n"); break;
		case -1010: printf("Error: *clBLAS* The memory object for Matrix B is too small\n"); break;
		case -1009: printf("Error: *clBLAS* The memory object for Matrix C is too small\n"); break;
		case -1008: printf("Error: *clBLAS* The memory object for Vector X is too small\n"); break;
		case -1007: printf("Error: *clBLAS* The memory object for Vector Y is too small\n"); break;
		case -1001: printf("Error: Code -1001: no GPU available?\n"); break;
		default: printf("Error: Unknown with code %d\n",error);break;
		}
		exit(1);
	}
}


bool CNN::Backward_output()
{
	/*
	init_variable(delta_neuron_output, 0.0, num_neuron_output_CNN);

	float dE_dy[num_neuron_output_CNN];
	init_variable(dE_dy, 0.0, num_neuron_output_CNN);
	loss_function_gradient(neuron_output, data_single_label, dE_dy, num_neuron_output_CNN); // 损失函数: mean squared error(均方差)

	// delta = dE/da = (dE/dy) * (dy/da)
	for (int i = 0; i < num_neuron_output_CNN; i++) {
		float dy_da[num_neuron_output_CNN];
		init_variable(dy_da, 0.0, num_neuron_output_CNN);

		dy_da[i] = activation_function_tanh_derivative(neuron_output[i]);
		delta_bias_output[i] = dot_product(dE_dy, dy_da, num_neuron_output_CNN);
	}
	*/
	///*
	//printf("Back output\n");
    //init_variable(delta_neuron_output, 0.0, num_neuron_output_CNN);
	//status  = clEnqueueWriteBuffer(cmdQueue, clbuf_delta_neuron_output, CL_FALSE, 0, sizeof(float)*num_neuron_output_CNN, delta_neuron_output, 0, NULL, NULL);
    //status |= clEnqueueWriteBuffer(cmdQueue, clbuf_data_single_label, CL_FALSE, 0, sizeof(float)*num_neuron_output_CNN, data_single_label, 0, NULL, NULL);
	//check_error(status);
	//printf("Write Over!\n");
    
       
	size_t globalWorkSize[1];   	
    globalWorkSize[0] = (size_t)num_neuron_output_CNN;
    
	status = clEnqueueNDRangeKernel(cmdQueue,
		kernel7,
		1,
		NULL,
		globalWorkSize,
		NULL,
		0,
		NULL,
		NULL);
    check_error(status);
    //clFinish(cmdQueue);   
    
    //clEnqueueReadBuffer(cmdQueue, clbuf_delta_bias_output, CL_TRUE, 0, sizeof(float)*num_neuron_output_CNN, delta_bias_output, 0, NULL, NULL);	

	//*/
	return true;
}

bool CNN::Backward_C5()
{
    /*
	init_variable(delta_neuron_C5, 0.0, num_neuron_C5_CNN);
	init_variable(delta_weight_output, 0.0, len_weight_output_CNN);
	//init_variable(delta_bias_output, 0.0, len_bias_output_CNN);

	for (int c = 0; c < num_neuron_C5_CNN; c++) {
		// propagate delta to previous layer
		// prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
		delta_bias_C5[c] = dot_product(&delta_bias_output[0], &weight_output[c * num_neuron_output_CNN], num_neuron_output_CNN);
		delta_bias_C5[c] *= activation_function_tanh_derivative(neuron_C5[c]);
		//printf("delta_neuron_C5, %d, %f\n", c, delta_neuron_C5[c]);
		// accumulate weight-step using delta
		// dW[c * out_size + i] += current_delta[i] * prev_out[c]
		muladd(&delta_bias_output[0], neuron_C5[c], num_neuron_output_CNN, &delta_weight_output[0] + c * num_neuron_output_CNN);
	}
	*/
	///*
	//printf("Back C5\n");
	//status  = clEnqueueWriteBuffer(cmdQueue, clbuf_delta_bias_output, CL_FALSE, 0, sizeof(float)*num_neuron_output_CNN, delta_bias_output, 0, NULL, NULL);
    //status |= clEnqueueWriteBuffer(cmdQueue, clbuf_weight_output, CL_FALSE, 0, sizeof(float)*len_weight_output_CNN, weight_output, 0, NULL,NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_neuron_C5, CL_FALSE,0,sizeof(float)*num_neuron_C5_CNN, data_single_label, 0, NULL, NULL);
	//check_error(status); 
	//printf("Write Over!\n");
     
    //kernel8  = clCreateKernel(program2, "back_c5", &status);
    //if(status) printf("Create kernel8 error\n");
    
 
    
	size_t globalWorkSize[1];
	size_t localWorkSize[1];

    globalWorkSize[0] = (size_t)num_neuron_C5_CNN * num_neuron_output_CNN << 1;
	localWorkSize[0] = (size_t)num_neuron_output_CNN;

	status = clEnqueueNDRangeKernel(cmdQueue,
		kernel8,
		1,
		NULL,
		globalWorkSize,
		localWorkSize,
		0,
		NULL,
		NULL);
    check_error(status);
    //clFinish(cmdQueue);
    
	//clEnqueueReadBuffer(cmdQueue,clbuf_delta_bias_C5,CL_TRUE,0,sizeof(float)*num_neuron_C5_CNN,delta_bias_C5,0,NULL,NULL);
	//clEnqueueReadBuffer(cmdQueue,clbuf_delta_weight_output,CL_TRUE,0,sizeof(float)*len_weight_output_CNN,delta_weight_output,0,NULL,NULL);	
    
	//*/
	return true;
}

bool CNN::Backward_S4()
{
	/*
	init_variable(delta_neuron_S4, 0.0, num_neuron_S4_CNN);
	init_variable(delta_weight_C5, 0.0, len_weight_C5_CNN);
	//init_variable(delta_bias_C5, 0.0, len_bias_C5_CNN);

	// propagate delta to previous layer
	// prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
	float temp1[num_neuron_S4_CNN] = {0};
	float temp2[len_weight_C5_CNN] = {0};
	for (int inc = 0; inc < num_map_S4_CNN; inc++) {
		for (int outc = 0; outc < num_map_C5_CNN; outc++) {
			int addr1 = get_index(0, 0, num_map_S4_CNN * outc + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_S4_CNN * num_map_C5_CNN);
			int addr2 = get_index(0, 0, outc, width_image_C5_CNN, height_image_C5_CNN, num_map_C5_CNN);
			int addr3 = get_index(0, 0, inc, width_image_S4_CNN, height_image_S4_CNN, num_map_S4_CNN);

			const float* pw = &weight_C5[0] + addr1;
			const float* pdelta_src = &delta_neuron_C5[0] + addr2;
			float* pdelta_dst = &temp1[0] + addr3;

			for (int y = 0; y < height_image_C5_CNN; y++) {
				for (int x = 0; x < width_image_C5_CNN; x++) {
					const float* ppw = pw;
					const float ppdelta_src = pdelta_src[y * width_image_C5_CNN + x];
					float* ppdelta_dst = pdelta_dst + y * width_image_S4_CNN + x;

					for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
						for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
							int addr = wy * width_image_S4_CNN + wx;
							ppdelta_dst[addr] += *ppw++ * ppdelta_src;
						}
					}	
				}
			}
		}
	}

	for (int i = 0; i < num_neuron_S4_CNN; i++) {
		temp1[i] *= activation_function_tanh_derivative(neuron_S4[i]);
	}

	// accumulate dw
	for (int inc = 0; inc < num_map_S4_CNN; inc++) {
		for (int outc = 0; outc < num_map_C5_CNN; outc++) {
			for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
				for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
					int addr1 = get_index(wx, wy, inc, width_image_S4_CNN, height_image_S4_CNN, num_map_S4_CNN);
					int addr2 = get_index(0, 0, outc, width_image_C5_CNN, height_image_C5_CNN, num_map_C5_CNN);
					int addr3 = get_index(wx, wy, num_map_S4_CNN * outc + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_S4_CNN * num_map_C5_CNN);

					float dst = 0.0;
					const float* prevo = &neuron_S4[0] + addr1;
					const float* delta = &delta_neuron_C5[0] + addr2;

					for (int y = 0; y < height_image_C5_CNN; y++) {
						dst += dot_product(prevo + y * width_image_S4_CNN, delta + y * width_image_C5_CNN, width_image_C5_CNN);
					}
					temp2[addr3] = dst;	
				}
			}
		}
	}
	*/
	
	///*
	//printf("Back S4\n");
	//status  = clEnqueueWriteBuffer(cmdQueue, clbuf_neuron_S4, CL_FALSE, 0, sizeof(float)*num_neuron_S4_CNN, neuron_S4, 0, NULL, NULL);
    //status |= clEnqueueWriteBuffer(cmdQueue, clbuf_weight_C5, CL_FALSE, 0, sizeof(float)*len_weight_C5_CNN, weight_C5, 0, NULL, NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_delta_bias_C5, CL_FALSE, 0, sizeof(float)*num_neuron_C5_CNN, delta_bias_C5, 0, NULL, NULL);
	//check_error(status); 
	//printf("Write Over!\n");
       
    
	size_t globalWorkSize[2];   	
    globalWorkSize[0] = (size_t)num_map_S4_CNN * num_neuron_C5_CNN * 2;
	globalWorkSize[1] = (size_t)width_kernel_conv_CNN * height_kernel_conv_CNN;
	size_t localWorkSize[2];
	localWorkSize[0] = (size_t)num_neuron_C5_CNN;
	localWorkSize[1] = (size_t)1;

    status = clEnqueueNDRangeKernel(cmdQueue,
		kernel9,
		2,
		NULL,
		globalWorkSize,
		localWorkSize,
		0,
		NULL,
		NULL);
    check_error(status);
    //clFinish(cmdQueue);

	//clEnqueueReadBuffer(cmdQueue, clbuf_delta_neuron_S4, CL_TRUE, 0, sizeof(float)*num_neuron_S4_CNN, delta_neuron_S4, 0, NULL, NULL);	
	//clEnqueueReadBuffer(cmdQueue, clbuf_delta_weight_C5, CL_TRUE, 0, sizeof(float)*len_weight_C5_CNN, delta_weight_C5, 0, NULL, NULL);	
    
	//*/
	/*
	for(int i = 0; i < num_neuron_S4_CNN; i++)
		if(temp1[i] - delta_neuron_S4[i] > 0.000001) {
			printf("S41\n");
			exit(0);
		}
	for(int i = 0; i < len_weight_C5_CNN; i++){
		if(temp2[i] - delta_weight_C5[i] > 0.000001) {
			printf("S42\n");
			exit(0);
		}
	}
	*/
	return true;
}

bool CNN::Backward_C3()
{
	/*
    init_variable(delta_neuron_C3, 0.0, num_neuron_C3_CNN);
	init_variable(delta_weight_S4, 0.0, len_weight_S4_CNN);
	//init_variable(delta_bias_S4, 0.0, len_bias_S4_CNN);

	float scale_factor = 1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);
	float temp1[num_neuron_C3_CNN];
	float temp2[len_weight_S4_CNN];

	for (int c = 0; c < num_map_C3_CNN; c++) {
		for (int y = 0; y < height_image_C3_CNN; y += height_kernel_pooling_CNN) {
			for (int x = 0; x < width_image_C3_CNN; x += width_kernel_pooling_CNN) {
				int dymax = min(size_pooling_CNN, height_image_C3_CNN - y);
				int dxmax = min(size_pooling_CNN, width_image_C3_CNN - x);
				int dstx = x / width_kernel_pooling_CNN;
				int dsty = y / height_kernel_pooling_CNN;

				for (int dy = 0; dy < dymax; dy++) {
					for (int dx = 0; dx < dxmax; dx++) {
						int index_in = get_index(x + dx, y + dy, c, width_image_C3_CNN, height_image_C3_CNN, num_map_C3_CNN);
						int index_out = get_index(dstx, dsty, c, width_image_S4_CNN, height_image_S4_CNN, num_map_S4_CNN);

						float delta = 0.0;
						delta += weight_S4[c] * delta_neuron_S4[index_out];
						temp1[index_in] = delta * scale_factor * activation_function_tanh_derivative(neuron_C3[index_in]);
					}
				}
			}
		}
	}

    for (int c = 0; c < num_map_C3_CNN; c++) {
    	float diff = 0;
		for (int y = 0; y < height_image_C3_CNN; y += height_kernel_pooling_CNN) {
			for (int x = 0; x < width_image_C3_CNN; x += width_kernel_pooling_CNN) {
				int dymax = min(size_pooling_CNN, height_image_C3_CNN - y);
				int dxmax = min(size_pooling_CNN, width_image_C3_CNN - x);
				int dstx = x / width_kernel_pooling_CNN;
				int dsty = y / height_kernel_pooling_CNN;

				for (int dy = 0; dy < dymax; dy++) {
					for (int dx = 0; dx < dxmax; dx++) {
						int index_in = get_index(x + dx, y + dy, c, width_image_C3_CNN, height_image_C3_CNN, num_map_C3_CNN);
						int index_out = get_index(dstx, dsty, c, width_image_S4_CNN, height_image_S4_CNN, num_map_S4_CNN);
						diff += neuron_C3[index_in] * delta_neuron_S4[index_out];
					}
				}
			}
		}
		temp2[c] = diff * scale_factor;
	}
	*/
	///*
	//printf("Back C3\n");
	//status  = clEnqueueWriteBuffer(cmdQueue, clbuf_delta_neuron_S4, CL_FALSE, 0, sizeof(float)*num_neuron_S4_CNN, delta_neuron_S4, 0, NULL, NULL);
    //status |= clEnqueueWriteBuffer(cmdQueue, clbuf_weight_S4, CL_FALSE, 0, sizeof(float)*len_weight_S4_CNN, weight_S4, 0, NULL, NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_neuron_C3, CL_FALSE, 0, sizeof(float)*num_neuron_S4_CNN, neuron_C3, 0, NULL, NULL);
	//check_error(status); 
	//printf("Write Over!\n");
 
    size_t globalWorkSize[2];   	
    globalWorkSize[0] = (size_t)num_map_C3_CNN * height_image_C3_CNN;
	globalWorkSize[1] = (size_t)width_image_C3_CNN * 2;
	size_t localWorkSize[2];
	localWorkSize[0] = (size_t)height_image_C3_CNN;
	localWorkSize[1] = (size_t)width_image_C3_CNN;
        
    status = clEnqueueNDRangeKernel(cmdQueue,
		kernel10,
		2,
		NULL,
		globalWorkSize,
		localWorkSize,
		0,
		NULL,
		NULL);
    //clFinish(cmdQueue); 
	check_error(status);

	//clEnqueueReadBuffer(cmdQueue, clbuf_delta_neuron_C3, CL_TRUE, 0, sizeof(float)*num_neuron_C3_CNN, delta_neuron_C3, 0, NULL, NULL);	
    //clEnqueueReadBuffer(cmdQueue, clbuf_delta_weight_S4, CL_TRUE, 0, sizeof(float)*len_weight_S4_CNN, delta_weight_S4, 0, NULL, NULL);	
	
	/*
	for(int i = 0; i < num_neuron_C3_CNN; i++)
		if(temp1[i] - delta_neuron_C3[i] > 0.000001){
			printf("C31\n");
			exit(0);
		}
	for(int i = 0; i < len_weight_S4_CNN; i++)
		if(temp2[i] - delta_weight_S4[i] > 0.000001){
			printf("C32\n");
			exit(0);
		}
	*/
	return true;
}

bool CNN::Backward_S2()
{
	/*
	init_variable(delta_neuron_S2, 0.0, num_neuron_S2_CNN);
	init_variable(delta_weight_C3, 0.0, len_weight_C3_CNN);
	//init_variable(delta_bias_C3, 0.0, len_bias_C3_CNN);
	float temp1[num_neuron_S2_CNN] = {0};
	float temp2[len_weight_C3_CNN] = {0};
	// propagate delta to previous layer
	for (int inc = 0; inc < num_map_S2_CNN; inc++) {
		for (int outc = 0; outc < num_map_C3_CNN; outc++) {
			if (!tbl[inc][outc]) continue;

			int addr1 = get_index(0, 0, num_map_S2_CNN * outc + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_S2_CNN * num_map_C3_CNN);
			int addr2 = get_index(0, 0, outc, width_image_C3_CNN, height_image_C3_CNN, num_map_C3_CNN);
			int addr3 = get_index(0, 0, inc, width_image_S2_CNN, height_image_S2_CNN, num_map_S2_CNN);

			const float *pw = &weight_C3[0] + addr1;
			const float *pdelta_src = &delta_neuron_C3[0] + addr2;;
			float* pdelta_dst = &temp1[0] + addr3;

			for (int y = 0; y < height_image_C3_CNN; y++) {
				for (int x = 0; x < width_image_C3_CNN; x++) {
					const float* ppw = pw;
					const float ppdelta_src = pdelta_src[y * width_image_C3_CNN + x];
					float* ppdelta_dst = pdelta_dst + y * width_image_S2_CNN + x;

					for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
						for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
							ppdelta_dst[wy * width_image_S2_CNN + wx] += *ppw++ * ppdelta_src;
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < num_neuron_S2_CNN; i++) {
		temp1[i] *= activation_function_tanh_derivative(neuron_S2[i]);
	}

	// accumulate dw
	for (int inc = 0; inc < num_map_S2_CNN; inc++) {
		for (int outc = 0; outc < num_map_C3_CNN; outc++) {
			if (!tbl[inc][outc]) continue;

			for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
				for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
					int addr1 = get_index(wx, wy, inc, width_image_S2_CNN, height_image_S2_CNN, num_map_S2_CNN);
					int addr2 = get_index(0, 0, outc, width_image_C3_CNN, height_image_C3_CNN, num_map_C3_CNN);
					int addr3 = get_index(wx, wy, num_map_S2_CNN * outc + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_S2_CNN * num_map_C3_CNN);
						
					float dst = 0.0;
					const float* prevo = &neuron_S2[0] + addr1;
					const float* delta = &delta_neuron_C3[0] + addr2;

					for (int y = 0; y < height_image_C3_CNN; y++) {
						dst += dot_product(prevo + y * width_image_S2_CNN, delta + y * width_image_C3_CNN, width_image_C3_CNN);
					}
					temp2[addr3] = dst;
				}
			}
		}
	}
	*/
   	///*
	//printf("Back S2\n");
	//status  = clEnqueueWriteBuffer(cmdQueue, clbuf_delta_neuron_C3, CL_FALSE, 0, sizeof(float)*num_neuron_C3_CNN, delta_neuron_C3, 0, NULL, NULL);
    //status |= clEnqueueWriteBuffer(cmdQueue, clbuf_weight_C3, CL_FALSE, 0, sizeof(float)*len_weight_C3_CNN, weight_C3, 0, NULL, NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_neuron_S2, CL_FALSE, 0, sizeof(float)*num_neuron_S2_CNN, neuron_S2, 0, NULL, NULL);
	//check_error(status); 
	//printf("Write Over!\n");
    
    
        
    size_t globalWorkSize[2];   	
    globalWorkSize[0] = (size_t)num_map_S2_CNN * (width_image_S2_CNN >> 1) * 10;
	globalWorkSize[1] = (size_t)4 * (height_image_S2_CNN >> 1);
	size_t localWorkSize[2];
	localWorkSize[0] = (size_t)(width_image_S2_CNN >> 1) * 10;
	localWorkSize[1] = (size_t)(height_image_S2_CNN >> 1);
     
    status = clEnqueueNDRangeKernel(cmdQueue,
		kernel11,
		2,
		NULL,
		globalWorkSize,
		localWorkSize,
		0,
		NULL,
		NULL);
    check_error(status);
    //clFinish(cmdQueue); 

	//clEnqueueReadBuffer(cmdQueue,clbuf_delta_neuron_S2,CL_TRUE,0,sizeof(float)*num_neuron_S2_CNN,delta_neuron_S2,0,NULL,NULL);
    
    //kernel12  = clCreateKernel(program2, "back2_s2", &status);
    //if(status) printf("Create kernel11 error\n");
    
   
           	
    globalWorkSize[0] = (size_t)num_map_S2_CNN * 10 * width_image_C3_CNN;
	globalWorkSize[1] = (size_t)width_kernel_conv_CNN * height_kernel_conv_CNN * height_image_C3_CNN;
	localWorkSize[0] = (size_t)width_image_C3_CNN * num_map_S2_CNN;
	localWorkSize[1] = (size_t)height_image_C3_CNN;
     
    status = clEnqueueNDRangeKernel(cmdQueue,
		kernel12,
		2,
		NULL,
		globalWorkSize,
		localWorkSize,
		0,
		NULL,
		NULL);
	check_error(status);
    //clFinish(cmdQueue);

	//clEnqueueReadBuffer(cmdQueue,clbuf_delta_weight_C3,CL_TRUE,0,sizeof(float)*len_weight_C3_CNN,delta_weight_C3,0,NULL,NULL);	
	
	//*/
	/*
	for(int i = 0; i < num_neuron_S2_CNN; i++)
		if(temp1[i] - delta_neuron_S2[i] > 0.000001){
			printf("S21\n");
			exit(0);
		}
	for(int i = 0; i < len_weight_C3_CNN; i++)
		if(temp2[i] - delta_weight_C3[i] > 0.000001){
			printf("S22\n");
			exit(0);
		}
	*/
	return true;
}

bool CNN::Backward_C1()
{   
	/*
   	init_variable(delta_neuron_C1, 0.0, num_neuron_C1_CNN);
	init_variable(delta_weight_S2, 0.0, len_weight_S2_CNN);
	//init_variable(delta_bias_S2, 0.0, len_bias_S2_CNN);
	
	float temp1[num_neuron_C1_CNN] = {0};
	float temp2[len_weight_S2_CNN] = {0};

	float scale_factor = 1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);

	for (int c = 0; c < num_map_C1_CNN; c++) {
		for (int y = 0; y < height_image_C1_CNN; y += height_kernel_pooling_CNN) {
			for (int x = 0; x < width_image_C1_CNN; x += width_kernel_pooling_CNN) {
				int dymax = min(size_pooling_CNN, height_image_C1_CNN - y);
				int dxmax = min(size_pooling_CNN, width_image_C1_CNN - x);
				int dstx = x / width_kernel_pooling_CNN;
				int dsty = y / height_kernel_pooling_CNN;

				for (int dy = 0; dy < dymax; dy++) {
					for (int dx = 0; dx < dxmax; dx++) {
						int index_in = get_index(x + dx, y + dy, c, width_image_C1_CNN, height_image_C1_CNN, num_map_C1_CNN);
						int index_out = get_index(dstx, dsty, c, width_image_S2_CNN, height_image_S2_CNN, num_map_S2_CNN);
						float delta = 0.0;
						delta += weight_S2[c] * delta_neuron_S2[index_out];
						temp1[index_in] = delta * scale_factor * activation_function_tanh_derivative(neuron_C1[index_in]);
					}
				}
			}
		}
	}

    for (int c = 0; c < num_map_C1_CNN; c++) {
    	float diff = 0.0;
		for (int y = 0; y < height_image_C1_CNN; y += height_kernel_pooling_CNN) {
			for (int x = 0; x < width_image_C1_CNN; x += width_kernel_pooling_CNN) {
				int dymax = min(size_pooling_CNN, height_image_C1_CNN - y);
				int dxmax = min(size_pooling_CNN, width_image_C1_CNN - x);
				int dstx = x / width_kernel_pooling_CNN;
				int dsty = y / height_kernel_pooling_CNN;

				for (int dy = 0; dy < dymax; dy++) {
					for (int dx = 0; dx < dxmax; dx++) {
						int index_in = get_index(x + dx, y + dy, c, width_image_C1_CNN, height_image_C1_CNN, num_map_C1_CNN);
						int index_out = get_index(dstx, dsty, c, width_image_S2_CNN, height_image_S2_CNN, num_map_S2_CNN);
						diff += neuron_C1[index_in] * delta_neuron_S2[index_out];
					}
				}
			}
		}
		temp2[c] = diff * scale_factor;
	}
	*/
	///*
	//printf("Back C1\n");
	//status  = clEnqueueWriteBuffer(cmdQueue, clbuf_delta_neuron_S2, CL_FALSE, 0, sizeof(float)*num_neuron_S2_CNN, delta_neuron_S2, 0, NULL, NULL);
    //status |= clEnqueueWriteBuffer(cmdQueue, clbuf_neuron_C1, CL_FALSE, 0, sizeof(float)*num_neuron_C1_CNN, neuron_C1, 0, NULL, NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_weight_S2, CL_FALSE, 0, sizeof(float)*len_weight_S2_CNN, weight_S2, 0, NULL, NULL);
	//check_error(status); 
	//printf("Write Over!\n");
    
    

    size_t globalWorkSize[2];
    globalWorkSize[0] = (size_t)num_map_C1_CNN * height_image_C1_CNN;
	globalWorkSize[1] = (size_t)width_image_C1_CNN * 2;
	size_t localWorkSize[2];
	localWorkSize[0] = (size_t)height_image_C1_CNN;
	localWorkSize[1] = (size_t)width_image_C1_CNN;
  
    status = clEnqueueNDRangeKernel(cmdQueue,
		kernel13,
		2,
		NULL,
		globalWorkSize,
		localWorkSize,
		0,
		NULL,
		NULL);
    check_error(status);
    //clFinish(cmdQueue);

	//clEnqueueReadBuffer(cmdQueue,clbuf_delta_neuron_C1,CL_TRUE,0,sizeof(float)*num_neuron_C1_CNN,delta_neuron_C1,0,NULL,NULL);	
    //clEnqueueReadBuffer(cmdQueue,clbuf_delta_weight_S2,CL_TRUE,0,sizeof(float)*len_weight_S2_CNN,delta_weight_S2,0,NULL,NULL);	
    
	//*/
	/*
	for(int i = 0; i < num_neuron_C1_CNN; i++)
		if(temp1[i] - delta_neuron_C1[i] > 0.000001){
			printf("C11\n");
			exit(0);
		}
	for(int i = 0; i < len_weight_S2_CNN; i++)
		if(temp2[i] - delta_weight_S2[i] > 0.000001){
			printf("C12\n");
			exit(0);
		}
	*/
	return true;
}

bool CNN::Backward_input()
{ 
	/*
   	init_variable(delta_neuron_input, 0.0, num_neuron_input_CNN);
	init_variable(delta_weight_C1, 0.0, len_weight_C1_CNN);
	//init_variable(delta_bias_C1, 0.0, len_bias_C1_CNN);

	float temp1[num_neuron_input_CNN] = {0};
	float temp2[len_weight_C1_CNN] = {0};
	// propagate delta to previous layer
	for (int inc = 0; inc < num_map_input_CNN; inc++) {
		for (int outc = 0; outc < num_map_C1_CNN; outc++) {
			int addr1 = get_index(0, 0, num_map_input_CNN * outc + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_C1_CNN);
			int addr2 = get_index(0, 0, outc, width_image_C1_CNN, height_image_C1_CNN, num_map_C1_CNN);
			int addr3 = get_index(0, 0, inc, width_image_input_CNN, height_image_input_CNN, num_map_input_CNN);

			const float* pw = &weight_C1[0] + addr1;
			const float* pdelta_src = &delta_neuron_C1[0] + addr2;
			float* pdelta_dst = &temp1[0] + addr3;

			for (int y = 0; y < height_image_C1_CNN; y++) {
				for (int x = 0; x < width_image_C1_CNN; x++) {
					const float* ppw = pw;
					const float ppdelta_src = pdelta_src[y * width_image_C1_CNN + x];
					float* ppdelta_dst = pdelta_dst + y * width_image_input_CNN + x;

					for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
						for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
							ppdelta_dst[wy * width_image_input_CNN + wx] += *ppw++ * ppdelta_src;
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < num_neuron_input_CNN; i++) {
		temp1[i] *= activation_function_identity_derivative(data_single_image[i]);
	}

	// accumulate dw
	for (int inc = 0; inc < num_map_input_CNN; inc++) {
		for (int outc = 0; outc < num_map_C1_CNN; outc++) {
			for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
				for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
					int addr1 = get_index(wx, wy, inc, width_image_input_CNN, height_image_input_CNN, num_map_input_CNN);
					int addr2 = get_index(0, 0, outc, width_image_C1_CNN, height_image_C1_CNN, num_map_C1_CNN);
					int addr3 = get_index(wx, wy, num_map_input_CNN * outc + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_C1_CNN);

					float dst = 0.0;
					const float* prevo = data_single_image + addr1;//&neuron_input[0]
					const float* delta = &delta_neuron_C1[0] + addr2;

					for (int y = 0; y < height_image_C1_CNN; y++) {
						dst += dot_product(prevo + y * width_image_input_CNN, delta + y * width_image_C1_CNN, width_image_C1_CNN);
					}

					temp2[addr3] += dst;
				}
			}
		}
	}
	*/
	///*
	//printf("Back input\n");
	//status  = clEnqueueWriteBuffer(cmdQueue, clbuf_delta_neuron_C1, CL_FALSE, 0, sizeof(float)*num_neuron_C1_CNN, delta_neuron_C1, 0, NULL, NULL);
    //status |= clEnqueueWriteBuffer(cmdQueue, clbuf_weight_C1, CL_FALSE, 0, sizeof(float)*len_weight_C1_CNN, weight_C1, 0, NULL, NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_data_single_image, CL_FALSE, 0, sizeof(float)*num_neuron_input_CNN, data_single_image, 0, NULL, NULL);
	//check_error(status); 
	//printf("Write Over!\n");
      
    
    size_t globalWorkSize[2];   	
    globalWorkSize[0] = (size_t)height_image_input_CNN * num_map_C1_CNN;
	globalWorkSize[1] = (size_t)width_image_input_CNN;
	size_t localWorkSize[2];
	localWorkSize[0] = (size_t)(height_image_input_CNN >> 2) * num_map_C1_CNN;
	localWorkSize[1] = (size_t)(width_image_input_CNN >> 1);
        
    status = clEnqueueNDRangeKernel(cmdQueue,
		kernel14,
		2,
		NULL,
		globalWorkSize,
		localWorkSize,
		0,
		NULL,
		NULL);
    check_error(status);
    //clFinish(cmdQueue);

	//clEnqueueReadBuffer(cmdQueue,clbuf_delta_neuron_input,CL_TRUE,0,sizeof(float)*num_neuron_input_CNN,delta_neuron_input,0,NULL,NULL);	
      
    globalWorkSize[0] = (size_t)num_map_C1_CNN *  height_image_C1_CNN;
	globalWorkSize[1] = (size_t)width_kernel_conv_CNN * width_kernel_conv_CNN * width_image_C1_CNN;
	localWorkSize[0] = (size_t)height_image_C1_CNN;
	localWorkSize[1] = (size_t)width_image_C1_CNN;
         
    status = clEnqueueNDRangeKernel(cmdQueue,
		kernel15,
		2,
		NULL,
		globalWorkSize,
		localWorkSize,
		0,
		NULL,
		NULL);
    check_error(status);
    clFinish(cmdQueue);

	//clEnqueueReadBuffer(cmdQueue,clbuf_delta_weight_C1,CL_TRUE,0,sizeof(float)*len_weight_C1_CNN,delta_weight_C1,0,NULL,NULL);	
    
	//*/
	/*
	for(int i = 0; i < num_neuron_input_CNN; i++)
		if(temp1[i] - delta_neuron_input[i] > 0.000001){
			printf("input1\n");
			exit(0);
		}
	for(int i = 0; i < len_weight_C1_CNN; i++)
		if(temp2[i] - delta_weight_C1[i] > 0.000001){
			printf("input2\n");
			exit(0);
		}
	*/
	return true;
}
