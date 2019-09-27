/*
 * forward.cpp
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

bool CNN::Forward_C1()
{
    //-----------------------------------------------------
    // STEP 6: Write host data to device buffers
    //-----------------------------------------------------     
    //status  = clEnqueueWriteBuffer(cmdQueue, clbuf_data_single_image, CL_FALSE,0,sizeof(float)*num_neuron_input_CNN,data_single_image,0,NULL,NULL);
    //status |= clEnqueueWriteBuffer(cmdQueue, clbuf_data_single_label, CL_FALSE,0,sizeof(float)*num_neuron_output_CNN,data_single_label,0,NULL,NULL);

    //status |= clEnqueueWriteBuffer(cmdQueue, clbuf_weight_C1, CL_FALSE,0,sizeof(float)*len_weight_C1_CNN,weight_C1,0,NULL,NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_bias_C1, CL_FALSE,0,sizeof(float)*len_bias_C1_CNN,bias_C1,0,NULL,NULL);
	//if(first) 
    //status |= clEnqueueWriteBuffer(cmdQueue, clbuf_neuron_C1, CL_FALSE,0,sizeof(float)*num_neuron_C1_CNN,neuron_C1,0,NULL,NULL);
    //if(status) printf("write to device buffer error\n");  
    //-----------------------------------------------------
    // STEP 10: Configure the work-item structure
    //-----------------------------------------------------    
    size_t globalWorkSize[3];	
    size_t localSize[3];
    localSize[0] = (size_t)1;
    localSize[1] = (size_t)BS1;
    localSize[2] = (size_t)BS1;
    globalWorkSize[0] = (size_t)num_map_C1_CNN;
    globalWorkSize[1] = (size_t)height_image_C1_CNN;///BY;
	globalWorkSize[2] = (size_t)width_image_C1_CNN;///BX;
    //-----------------------------------------------------
    // STEP 11: Enqueue the kernel for execution
    //-----------------------------------------------------        
    status = clEnqueueNDRangeKernel(cmdQueue,kernel1,3,NULL,globalWorkSize,localSize,0,NULL,NULL);
    //status = clEnqueueNDRangeKernel(cmdQueue,kernel1,3,NULL,globalWorkSize,NULL,0,NULL,NULL);
    
    if(status) cout<<status<<" "<<"running Kernel1 error!\n";
    //clFinish(cmdQueue);
    //----------------------------------------------------- 
    // STEP 12: Read the output buffer back to the host
    //-----------------------------------------------------  
    //clEnqueueReadBuffer(cmdQueue,clbuf_neuron_C1,CL_TRUE,0,sizeof(float)*num_neuron_C1_CNN,neuron_C1,0,NULL,NULL);	
	//printf("\nneuron_C1: ");
    //for(int i = 0;i < num_neuron_C1_CNN;++ i) printf("%.3f ",neuron_C1[i]);cout<<endl;
    /*check
    float *temp = new float[num_neuron_C1_CNN];
	for (int channel = 0; channel < num_map_C1_CNN; channel++) {
		for (int y = 0; y < height_image_C1_CNN; y++) {
			for (int x = 0; x < width_image_C1_CNN; x++) {
				int index = (channel*height_image_C1_CNN*width_image_C1_CNN) + y*width_image_C1_CNN + x;  //当前神经元
				temp[index] = 0.0;
				//卷积运算
				for (int inc = 0; inc < num_map_input_CNN; inc++) {
					int addr1 = get_index(0, 0, num_map_input_CNN * channel + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_C1_CNN * num_map_input_CNN);
					int addr2 = get_index(0, 0, inc, width_image_input_CNN, height_image_input_CNN, num_map_input_CNN);
					const float* pw = &weight_C1[0] + addr1;       //卷积核
					const float* pi = data_single_image + addr2;   //输入图像
					float sum = 0.0;
					const float* ppw = pw;
					const float* ppi = pi + y * width_image_input_CNN + x;
					for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
						for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
							sum += *ppw++ * ppi[wy * width_image_input_CNN + wx];
						}
					}
					temp[index] += sum;
				}
				temp[index] += bias_C1[channel];     //加偏置
				temp[index] = activation_function_tanh(temp[index]);  //激励函数
			}
		}
	}
    //printf("temp: ");
    //for(int i = 0;i < num_neuron_C1_CNN;++ i) printf("%.3f ",temp[i]);cout<<endl;
    bool f = 1;
    for(int i = 0;i < num_neuron_C1_CNN;++ i) if(dcmp(neuron_C1[i]-temp[i])) {f = 0;cout<<i<<" "<<delta_neuron_C1[i]<<" "<<temp[i]<<" ";break;}
    puts(f?"forward_C1 is correct!":"forward_C1 is wrong!");
    delete []temp;
    */
	return true;
	
}

//ave 过程 乘以权值weight[i] | 再加上bias[i] | 最后sigmoid
bool CNN::Forward_S2()
{
	//-----------------------------------------------------
    // STEP 6: Write host data to device buffers
    //-----------------------------------------------------     
    //status  = clEnqueueWriteBuffer(cmdQueue, clbuf_weight_S2, CL_FALSE,0,sizeof(float)*len_weight_S2_CNN,weight_S2,0,NULL,NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_bias_S2, CL_FALSE,0,sizeof(float)*len_bias_S2_CNN,bias_S2,0,NULL,NULL);
	//if(first) 
    //status |= clEnqueueWriteBuffer(cmdQueue, clbuf_neuron_S2, CL_FALSE,0,sizeof(float)*num_neuron_S2_CNN,neuron_S2,0,NULL,NULL);
    //if(status) printf("write to device buffer error\n");
    //-----------------------------------------------------
    // STEP 10: Configure the work-item structure
    //----------------------------------------------------- 
    size_t globalWorkSize[3]; 
    size_t localSize[3];
    localSize[0] = (size_t)1;
    localSize[1] = (size_t)BS2;
    localSize[2] = (size_t)BS2;    	
    globalWorkSize[0] = (size_t)num_map_S2_CNN;
    globalWorkSize[1] = (size_t)height_image_S2_CNN;
	globalWorkSize[2] = (size_t)width_image_S2_CNN;
    //-----------------------------------------------------
    // STEP 11: Enqueue the kernel for execution
    //-----------------------------------------------------        
    status = clEnqueueNDRangeKernel(cmdQueue,kernel2,3,NULL,globalWorkSize,localSize,0,NULL,NULL);
    //status = clEnqueueNDRangeKernel(cmdQueue,kernel2,3,NULL,globalWorkSize,NULL,0,NULL,NULL);
  
    if(status) cout<<"running Kernel error!\n"; 
    //clFinish(cmdQueue);    
    //----------------------------------------------------- 
    // STEP 12: Read the output buffer back to the host
    //-----------------------------------------------------   
    //clEnqueueReadBuffer(cmdQueue,clbuf_neuron_S2,CL_TRUE,0,sizeof(float)*num_neuron_S2_CNN,neuron_S2,0,NULL,NULL);
	/*check
    float *temp = new float[num_neuron_S2_CNN];
    float scale_factor = 1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);//2*2=4
	for (int i=0; i<num_map_S2_CNN; i++) {
		int block = width_image_C1_CNN * height_image_C1_CNN * i;
		for (int y=0; y<height_image_S2_CNN; y++) {
			for (int x=0; x<width_image_S2_CNN; x++) {
				int rows = y * width_kernel_pooling_CNN;
				int cols = x * height_kernel_pooling_CNN;
				int index = (i*height_image_S2_CNN*width_image_S2_CNN) + y*width_image_S2_CNN + x;
				//ave pool
                temp[index] = 0.0;
				for (int m = 0; m < width_kernel_pooling_CNN; m++) {
					for (int n = 0; n < height_kernel_pooling_CNN; n++) {
                        temp[index] += weight_S2[i] * neuron_C1[(rows + m) * width_image_C1_CNN + cols + n + block];
					}
				}
				//
				temp[index] *= scale_factor;
				temp[index] += bias_S2[i] ;
				temp[index] = activation_function_tanh(temp[index]);//tanh激励函数
			}
		}
	}
    bool f = 1;
    for(int i = 0;i < num_neuron_S2_CNN;++ i) if(dcmp(neuron_S2[i]-temp[i])) {f = 0;break;}
    puts(f?"forward_S2 is correct!":"forward_S2 is wrong!");
    delete []temp;
    */
    return true;
}

bool CNN::Forward_C3()
{
	//-----------------------------------------------------
    // STEP 6: Write host data to device buffers
    //-----------------------------------------------------     
    //status  = clEnqueueWriteBuffer(cmdQueue, clbuf_weight_C3, CL_FALSE,0,sizeof(float)*len_weight_C3_CNN,weight_C3,0,NULL,NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_bias_C3, CL_FALSE,0,sizeof(float)*len_bias_C3_CNN,bias_C3,0,NULL,NULL);
	//if(first) 
    //status |= clEnqueueWriteBuffer(cmdQueue, clbuf_neuron_C3, CL_FALSE,0,sizeof(float)*num_neuron_C3_CNN,neuron_C3,0,NULL,NULL);
	//if(first) 
    //status |= clEnqueueWriteBuffer(cmdQueue, clbuf_tbl, CL_FALSE,0,sizeof(bool)*6*16,tbl,0,NULL,NULL);
    //if(status) printf("write to device buffer error\n");
    //-----------------------------------------------------
    // STEP 10: Configure the work-item structure
    //----------------------------------------------------- 
    size_t globalWorkSize[3];     
    size_t localSize[3];
    localSize[0] = (size_t)1;
    localSize[1] = (size_t)BS3;
    localSize[2] = (size_t)BS3;
    globalWorkSize[0] = (size_t)num_map_C3_CNN;
    globalWorkSize[1] = (size_t)height_image_C3_CNN;///BY;
	globalWorkSize[2] = (size_t)width_image_C3_CNN;///BX;
    //-----------------------------------------------------
    // STEP 11: Enqueue the kernel for execution
    //-----------------------------------------------------        
    status = clEnqueueNDRangeKernel(cmdQueue,kernel3,3,NULL,globalWorkSize,localSize,0,NULL,NULL);
   //status = clEnqueueNDRangeKernel(cmdQueue,kernel3,3,NULL,globalWorkSize,NULL,0,NULL,NULL);
  
    if(status) cout<<"running Kernel error!\n";    
    //clFinish(cmdQueue); 
    //----------------------------------------------------- 
    // STEP 12: Read the output buffer back to the host
    //-----------------------------------------------------   
    //clEnqueueReadBuffer(cmdQueue,clbuf_neuron_C3,CL_TRUE,0,sizeof(float)*num_neuron_C3_CNN,neuron_C3,0,NULL,NULL);
	//printf("neuron_C3: ");
    //for(int i = 0;i < 200;++ i) printf("%.3f ",neuron_C3[i]);cout<<endl;
    /*check
    float *temp = new float[num_neuron_C3_CNN];
    for (int channel = 0; channel < num_map_C3_CNN; channel++) {
		for (int y = 0; y < height_image_C3_CNN; y++) {
			for (int x = 0; x < width_image_C3_CNN; x++) {
				int index = (channel*height_image_C3_CNN*width_image_C3_CNN) + y*width_image_C3_CNN + x;  //当前神经元
				temp[index] = 0.0;
				//卷积运算
				for (int inc = 0; inc < num_map_S2_CNN; inc++) {
					if (!tbl[inc][channel]) continue;
					int addr1 = get_index(0, 0, num_map_S2_CNN * channel + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_C3_CNN * num_map_S2_CNN);
					int addr2 = get_index(0, 0, inc, width_image_S2_CNN, height_image_S2_CNN, num_map_S2_CNN);  //输入图像
					const float* pw = &weight_C3[0] + addr1;   //卷积核
					const float* pi = &neuron_S2[0] + addr2;   //输入图像
					float sum = 0.0;
					const float* ppw = pw;
					const float* ppi = pi + y * width_image_S2_CNN + x;
					for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
						for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
							sum += *ppw++ * ppi[wy * width_image_S2_CNN + wx];
						}
					}
					temp[index] += sum;
				}
				temp[index] += bias_C3[channel];     //加偏置
				temp[index] = activation_function_tanh(temp[index]);  //激励函数
			}
		}
	}
    //printf("temp: ");
    //for(int i = 0;i < 200;++ i) printf("%.3f ",temp[i]);cout<<endl;
    bool f = 1;
    for(int i = 0;i < num_neuron_C3_CNN;++ i) if(dcmp(neuron_C3[i]-temp[i])) {f = 0;break;}
    puts(f?"forward_C3 is correct!":"forward_C3 is wrong!");
    delete []temp;
    */
	return true;
}

bool CNN::Forward_S4()
{
    //-----------------------------------------------------
    // STEP 6: Write host data to device buffers
    //-----------------------------------------------------     
    //status = clEnqueueWriteBuffer(cmdQueue, clbuf_data_single_image, CL_FALSE,0,sizeof(float)*num_neuron_input_CNN,data_single_image,0,NULL,NULL);
    //status  = clEnqueueWriteBuffer(cmdQueue, clbuf_weight_S4, CL_FALSE,0,sizeof(float)*len_weight_S4_CNN,weight_S4,0,NULL,NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_bias_S4, CL_FALSE,0,sizeof(float)*len_bias_S4_CNN,bias_S4,0,NULL,NULL);
	//if(first) 
    //status |= clEnqueueWriteBuffer(cmdQueue, clbuf_neuron_S4, CL_FALSE,0,sizeof(float)*num_neuron_S4_CNN,neuron_S4,0,NULL,NULL);
    //if(status) printf("write to device buffer error\n");
    //-----------------------------------------------------
    // STEP 10: Configure the work-item structure
    //----------------------------------------------------- 
    size_t globalWorkSize[3];     	
    size_t localSize[3];
    localSize[0] = (size_t)1;
    localSize[1] = (size_t)BS4;
    localSize[2] = (size_t)BS4;    	
    globalWorkSize[0] = (size_t)num_map_S4_CNN;
    globalWorkSize[1] = (size_t)height_image_S4_CNN;
	globalWorkSize[2] = (size_t)width_image_S4_CNN;
    //-----------------------------------------------------
    // STEP 11: Enqueue the kernel for execution
    //-----------------------------------------------------        
    //status = clEnqueueNDRangeKernel(cmdQueue,kernel4,3,NULL,globalWorkSize,NULL,0,NULL,NULL);
    status = clEnqueueNDRangeKernel(cmdQueue,kernel4,3,NULL,globalWorkSize,localSize,0,NULL,NULL);
  
    if(status) cout<<"running Kernel error!\n";  
    //clFinish(cmdQueue);   
    //----------------------------------------------------- 
    // STEP 12: Read the output buffer back to the host
    //-----------------------------------------------------   
    //clEnqueueReadBuffer(cmdQueue,clbuf_neuron_S4,CL_TRUE,0,sizeof(float)*num_neuron_S4_CNN,neuron_S4,0,NULL,NULL);
	/*check
    float *temp = new float[num_neuron_S4_CNN];
	float scale_factor = 1.0 / (width_kernel_pooling_CNN * height_kernel_pooling_CNN);
	for (int i=0; i<num_map_S4_CNN; i++) {
		int block = width_image_C3_CNN * height_image_C3_CNN * i;
		for (int y=0; y<height_image_S4_CNN; y++) {
			for (int x=0; x<width_image_S4_CNN; x++) {
				int rows = y * width_kernel_pooling_CNN;
				int cols = x * height_kernel_pooling_CNN;
				int index = (i*height_image_S4_CNN*width_image_S4_CNN) + y*width_image_S4_CNN + x;

                temp[index] = 0.0;
				for (int m = 0; m < width_kernel_pooling_CNN; m++) {
					for (int n = 0; n < height_kernel_pooling_CNN; n++) {
                        temp[index] += weight_S4[i] * neuron_C3[(rows + m) * width_image_C3_CNN + cols + n + block];
					}
				}
				//
				temp[index] *= scale_factor;
				temp[index] += bias_S4[i] ;
				temp[index] = activation_function_tanh(temp[index]);
			}
		}
	}
    bool f = 1;
    for(int i = 0;i < num_neuron_S4_CNN;++ i) if(dcmp(neuron_S4[i]-temp[i])) {f = 0;break;}
    puts(f?"forward_S4 is correct!":"forward_S4 is wrong!");
    delete []temp;
    */
	return true;
}

bool CNN::Forward_C5()
{
    //-----------------------------------------------------
    // STEP 6: Write host data to device buffers
    //-----------------------------------------------------     
    //status  = clEnqueueWriteBuffer(cmdQueue, clbuf_weight_C5, CL_FALSE,0,sizeof(float)*len_weight_C5_CNN,weight_C5,0,NULL,NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_bias_C5, CL_FALSE,0,sizeof(float)*len_bias_C5_CNN,bias_C5,0,NULL,NULL);
	//if(first) 
    //    status |= clEnqueueWriteBuffer(cmdQueue, clbuf_neuron_C5, CL_FALSE,0,sizeof(float)*num_neuron_C5_CNN,neuron_C5,0,NULL,NULL);
    //if(status) printf("write to device buffer error\n");
    //-----------------------------------------------------
    // STEP 10: Configure the work-item structure
    //----------------------------------------------------- 
    size_t globalWorkSize[3];  
    size_t localSize[3];   
    localSize[0] = (size_t)1;
    localSize[1] = (size_t)BS5;
    localSize[2] = (size_t)BS5;
    globalWorkSize[0] = (size_t)num_map_C5_CNN;
    globalWorkSize[1] = (size_t)height_image_C5_CNN;
	globalWorkSize[2] = (size_t)width_image_C5_CNN;
    //-----------------------------------------------------
    // STEP 11: Enqueue the kernel for execution
    //-----------------------------------------------------        
    //status = clEnqueueNDRangeKernel(cmdQueue,kernel5,3,NULL,globalWorkSize,NULL,0,NULL,NULL);
    status = clEnqueueNDRangeKernel(cmdQueue,kernel5,3,NULL,globalWorkSize,localSize,0,NULL,NULL);
 
    if(status) cout<<"running Kernel error!\n"; 
    //clFinish(cmdQueue);    
    //----------------------------------------------------- 
    // STEP 12: Read the output buffer back to the host
    //-----------------------------------------------------   
    //clEnqueueReadBuffer(cmdQueue,clbuf_neuron_C5,CL_TRUE,0,sizeof(float)*num_neuron_C5_CNN,neuron_C5,0,NULL,NULL);
	/*check
    float *temp = new float[num_neuron_C5_CNN];
	for (int channel = 0; channel < num_map_C5_CNN; channel++) {
		for (int y = 0; y < height_image_C5_CNN; y++) {
			for (int x = 0; x < width_image_C5_CNN; x++) {
				int index = (channel*height_image_C5_CNN*width_image_C5_CNN) + y*width_image_C5_CNN + x;  //当前神经元
				temp[index] = 0.0;
				//卷积运算
				for (int inc = 0; inc < num_map_S4_CNN; inc++) {
					int addr1 = get_index(0, 0, num_map_S4_CNN * channel + inc, width_kernel_conv_CNN, height_kernel_conv_CNN, num_map_C5_CNN * num_map_S4_CNN);
					int addr2 = get_index(0, 0, inc, width_image_S4_CNN, height_image_S4_CNN, num_map_S4_CNN);
					const float* pw = &weight_C5[0] + addr1;       //卷积核
					const float* pi = &neuron_S4[0] + addr2;   //输入图像
					float sum = 0.0;
					const float* ppw = pw;
					const float* ppi = pi + y * width_image_S4_CNN + x;
					for (int wy = 0; wy < height_kernel_conv_CNN; wy++) {
						for (int wx = 0; wx < width_kernel_conv_CNN; wx++) {
							sum += *ppw++ * ppi[wy * width_image_S4_CNN + wx];
						}
					}
					temp[index] += sum;
				}
				temp[index] += bias_C5[channel];     //加偏置
				temp[index] = activation_function_tanh(temp[index]);  //激励函数
			}
		}
	}
    bool f = 1;
    for(int i = 0;i < num_neuron_C5_CNN;++ i) if(dcmp(neuron_C5[i]-temp[i])) {f = 0;break;}
    puts(f?"forward_C5 is correct!":"forward_C5 is wrong!");
    delete []temp;
    */
	return true;
}

bool CNN::Forward_output()
{
    //-----------------------------------------------------
    // STEP 6: Write host data to device buffers
    //-----------------------------------------------------     
    //status  = clEnqueueWriteBuffer(cmdQueue, clbuf_weight_output, CL_FALSE,0,sizeof(float)*len_weight_output_CNN,weight_output,0,NULL,NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_bias_output, CL_FALSE,0,sizeof(float)*len_bias_output_CNN,bias_output,0,NULL,NULL);
	//if(first) 
    //status |= clEnqueueWriteBuffer(cmdQueue, clbuf_neuron_output, CL_FALSE,0,sizeof(float)*num_neuron_output_CNN,neuron_output,0,NULL,NULL);
    //if(status) printf("write to device buffer error\n"); 
    //-----------------------------------------------------
    // STEP 10: Configure the work-item structure
    //----------------------------------------------------- 
    size_t globalWorkSize[1];     	
    globalWorkSize[0] = (size_t)num_map_output_CNN;
    size_t localSize[1];
    localSize[0] = (size_t)num_map_output_CNN;
    //-----------------------------------------------------
    // STEP 11: Enqueue the kernel for execution
    //-----------------------------------------------------        
    //status = clEnqueueNDRangeKernel(cmdQueue,kernel6,1,NULL,globalWorkSize,NULL,0,NULL,NULL);
    status = clEnqueueNDRangeKernel(cmdQueue,kernel6,1,NULL,globalWorkSize,localSize,0,NULL,NULL);

    if(status) cout<<"running Kernel error!\n";    
    clFinish(cmdQueue); 
    //----------------------------------------------------- 
    // STEP 12: Read the output buffer back to the host
    //-----------------------------------------------------   
    //clEnqueueReadBuffer(cmdQueue,clbuf_neuron_output,CL_TRUE,0,sizeof(float)*num_neuron_output_CNN,neuron_output,0,NULL,NULL);
    /*check
    float *temp = new float[num_neuron_output_CNN];
	for (int i = 0; i < num_neuron_output_CNN; i++) {
		temp[i] = 0.0;
		for (int c = 0; c < num_neuron_C5_CNN; c++) {
			temp[i] += weight_output[c * num_neuron_output_CNN + i] * neuron_C5[c];
		}
		temp[i] += bias_output[i];
		temp[i] = activation_function_tanh(temp[i]);
	}
    bool f = 1;
    for(int i = 0;i < num_neuron_output_CNN;++ i) if(dcmp(neuron_output[i]-temp[i])) {f = 0;break;}
    puts(f?"forward_output is correct!":"forward_output is wrong!");
    delete []temp;
    */
	return true;
}





