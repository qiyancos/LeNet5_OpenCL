//前向传播的opencl内核
#define len_weight_C1_CNN		150   //C1层权值数，5*5*6*1=150
#define len_bias_C1_CNN			6     //C1层阈值数，6
#define len_weight_S2_CNN		6     //S2层权值数,1*6=6
#define len_bias_S2_CNN			6     //S2层阈值数,6
#define len_weight_C3_CNN		2400  //C3层权值数，5*5*16*6=2400
#define len_bias_C3_CNN			16    //C3层阈值数,16
#define len_weight_S4_CNN		16    //S4层权值数，1*16=16
#define len_bias_S4_CNN			16    //S4层阈值数，16
#define len_weight_C5_CNN		48000 //C5层权值数，5*5*16*120=48000
#define len_bias_C5_CNN			120   //C5层阈值数，120
#define len_weight_output_CNN	1200  //输出层权值数，120*10=1200
#define len_bias_output_CNN		10    //输出层阈值数，10
// 特征图数量   feature maps
#define num_map_input_CNN		1 //输入层map个数
#define num_map_C1_CNN			6 //C1层map个数
#define num_map_S2_CNN			6 //S2层map个数
#define num_map_C3_CNN			16 //C3层map个数
#define num_map_S4_CNN			16 //S4层map个数
#define num_map_C5_CNN			120 //C5层map个数
#define num_map_output_CNN		10 //输出层map个数

#define BS1 7
#define BS2 14
#define BS3 5
#define BS4 5
#define BS5 1
#define filtersize 5
#define convsize 2
#define BX 2
#define BY 1


//===============================================================================//
//		四.以下是constant memory + local memory + workitem 优化的forward()内核函数
/*================================================================================//
 __kernel void  kernel_forward_c1(__global float *in,//data_single_image->data_input_train
                      __constant float  *weight,//weight_C1 卷积核
                      __constant float  *bias,//每个特征图有自己的bias_C1[len_bias_C1_CNN]
                      __global float  *out,//neuron_C1[] 
                      int channel,//num_map_C1_CNN 特征图的数量 6 
                      int out_width,//width_image_c1_CNN 28
                      int out_height,//height_image_c1_CNN 28
                      int kernel_width,//卷积核 5
					  int kernel_height,//卷积核 5
					  int in_num,//num_map_input_CNN 特征图的数量
					  int in_width,//width_image_input_CNN 32 
                      int in_height//height_image_input_CNN 32
					  )
{//每个神经元节点输出是并行的
	 channel = get_global_id(0);//out number map
    int  y = get_global_id(1);//  y/OY1
    int  x = get_global_id(2);//  x/OX1
		    
        int tidy=get_local_id(1);//[0,14)
        int tidx=get_local_id(2);//[0,14)

        float local pixel[num_map_input_CNN][BS1*BY+filtersize-1][BS1*BX+filtersize-1];//18,32

        for (int k=0; k<in_num; k++)
        {
            int addr2 = k*in_width*in_height;
			for(int i = 0;i < BX;++ i){
				for(int j = 0;j < BY;++ j){
					pixel[k][tidy+BS1*j][tidx+BS1*i]=in[addr2 + (y+BS1*j)*in_width + x +BS1*i];
				}
			}
			if(tidx < filtersize -1 ){
				for(int j = 0;j < BY;++ j)
					pixel[k][tidy+BS1*j][tidx+BS1*BX] = in[addr2+(y+j*BS1)*in_width+x+BS1*BX];
            }
			if(tidy < filtersize -1 ){
				for(int i = 0;i < BX;++ i)
					pixel[k][tidy+BS1*BY][tidx+BS1*i] = in[addr2+(y+BS1*BY)*in_width+x+BS1*i];
			}
			if(tidx < filtersize -1 &&tidy < filtersize-1){
			   	pixel[k][tidy+BS1*BY][tidx+BS1*BX] = in[addr2+(y+BS1*BY)*in_width+x+BS1*BX];
			}	
		}
        barrier(CLK_LOCAL_MEM_FENCE);
		
        int  index = (channel*out_height*out_width) + y*out_width + x;
        float sum[BX*BY] = {0.0};
		//float sum = 0.0;
		
		int inc = 0;
		int wx = 0;
		int wy = 0;
		out[index] = 0.0;
        for (inc=0; inc<in_num; inc++)
    {
		int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = inc*(BS1*BY+filtersize-1)*(BS1*BX+filtersize-1);
		for(int i = 0;i < BX;++ i)
			for(int j = 0;j < BY;++ j)
				sum[i+j*BX] = 0.0;
		__constant const float* pw = weight + addr1;   //卷积核,默认是__private变量
		__local const float* pi = &pixel[0][0][0] + addr2;       //输入图像
		__constant const float* ppw = pw;//卷积核
		
		__local const float* ppi = pi + tidy* BY * (BS1*BX+filtersize-1) + tidx * BX;//输入图像

		
        for(wy = 0; wy < kernel_height; wy++){
            for(wx = 0; wx < kernel_width; wx++){
					float filterItem = ppw[wy*kernel_width+wx];
					for(int i = 0;i < BX;++ i){
						for(int j = 0;j < BY;++ j){

							sum[i+j*BX] += filterItem * ppi[(wy+j)*(BS1*BX+filtersize-1)+wx+i];

						}
					}
            }
        }
		/*
		for(int i = 0;i < BX;++ i){
			for(int j = 0;j < BY;++ j){		
				out[index + tidx*(BX-1)+i + (tidy*(BY-1)+j)*out_width] += sum[i+j*BX];
			}
		}
		
	}
	
		for(int i = 0;i < BX;++ i){
			for(int j = 0;j < BY;++ j){

				sum[j*BX + i] += bias[channel];
        		out[index + tidx*(BX-1)+i + (tidy*(BY-1)+j)*out_width] = tanh((float)sum[j*BX + i]);
			}
		}
       
		
}
*/
__kernel void  kernel_forward_s2(__global float *in,//neuron_C1
					  constant float  *weight,//weight_S2[] 
                      constant float  *bias,//每个特征图有自己的bias_C2[len_bias_C2_CNN]
                      __global float  *out,//neuron_S2[] 
                      int channel,//num_map_C2_CNN 特征图的数量 6 
                      int out_width,//width_image_C2_CNN 14
                      int out_height,//height_image_C2_CNN 14
                      int kernel_width, //池化核 2
					  int kernel_height,//池化核 2
					  int in_num,//num_map_C1_CNN 特征图的数量 6
					  int in_width,//width_image_C1_CNN 28
                      int in_height//height_image_C1_CNN 28
					  )
{
	channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
	//local mem
	int tidy=get_local_id(1);
    int tidx=get_local_id(2);
	float local pixel[num_map_C1_CNN][BS2<<1][BS2<<1];
	pixel[channel][tidy][tidx] = in[channel*in_width*in_height+y*in_width+x];
	pixel[channel][tidy+BS2][tidx] = in[channel*in_width*in_height+(y+BS2)*in_width+x];
	pixel[channel][tidy][tidx+BS2] = in[channel*in_width*in_height+y*in_width+x+BS2];
	pixel[channel][tidy+BS2][tidx+BS2] = in[channel*in_width*in_height+(y+BS2)*in_width+x+BS2];
	barrier(CLK_LOCAL_MEM_FENCE);
	//
    //float scale_factor = 1.0 / (kernel_width * kernel_height);
    //int block =	 channel;
    int rows = tidy * kernel_width;
	int cols = tidx * kernel_height;
	int index = (channel*out_height*out_width) + y*out_width + x;

	out[index] = 0.0;
	for (int m = 0; m < kernel_width; m++) {
		for (int n = 0; n < kernel_height; n++) {
            out[index] += weight[channel] * pixel[channel][rows + m][cols + n];
		}
	}
	out[index] *= 0.25;  //scale_factor 池化层是2*2的 1/4;
	out[index] += bias[channel] ;
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_c3(__global float *in,//neuron_S2[]
                      constant float  *weight,//weight_C3[]
                      constant float  *bias,//bias_C3[]
                      __global float  *out,//neuron_C3[]
                      int channel,//num_map_C3_CNN 特征图的数量 16 
                      int out_width,//width_image_c3_CNN 10
                      int out_height,//height_image_c3_CNN 10
                      int kernel_width,//卷积核 5
					  int kernel_height,//卷积核 5
					  int in_num,//num_map_S2_CNN 特征图的数量6
					  int in_width,//width_image_S2_CNN 14 
                      int in_height,//height_image_S2_CNN 14
                      __global bool  *tbl //bool tbl[6][16]
					  )
{
        /* 
	 	channel = get_global_id(0);
    	int  y = get_global_id(1);
   	 	int  x = get_global_id(2);
		//local mem
		int tidy=get_local_id(1);
   		int tidx=get_local_id(2);
		float local pixel[num_map_S2_CNN][BS3*BY+filtersize-1][BS3*BX+filtersize-1];//18,32

        for (int k=0; k<in_num; k++)
        {
            int addr2 = k*in_width*in_height;
			for(int i = 0;i < BX;++ i){
				for(int j = 0;j < BY;++ j){
					pixel[k][tidy+BS3*j][tidx+BS3*i]=in[addr2 + (y+BS3*j)*in_width + x +BS3*i];
				}
			}
			if(tidx < filtersize -1 ){
				for(int j = 0;j < BY;++ j)
					pixel[k][tidy+BS3*j][tidx+BS3*BX] = in[addr2+(y+j*BS3)*in_width+x+BS3*BX];
            }
			if(tidy < filtersize -1 ){
				for(int i = 0;i < BX;++ i)
					pixel[k][tidy+BS3*BY][tidx+BS3*i] = in[addr2+(y+BS3*BY)*in_width+x+BS3*i];
			}
			if(tidx < filtersize -1 &&tidy < filtersize-1){
			   	pixel[k][tidy+BS3*BY][tidx+BS3*BX] = in[addr2+(y+BS3*BY)*in_width+x+BS3*BX];
			}	
		}
        barrier(CLK_LOCAL_MEM_FENCE);
		
        int  index = (channel*out_height*out_width) + y*out_width + x;
        float sum[BX*BY] = {0.0};
		//float sum = 0.0;
		
		int inc = 0;
		int wx = 0;
		int wy = 0;
		out[index] = 0.0;
        for (inc=0; inc<in_num; inc++)
    {
		if (!tbl[inc*16+channel]) continue;
		int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = inc*(BS3*BY+filtersize-1)*(BS3*BX+filtersize-1);
		for(int i = 0;i < BX;++ i)
			for(int j = 0;j < BY;++ j)
				sum[i+j*BX] = 0.0;
		__constant const float* pw = weight + addr1;   //卷积核,默认是__private变量
		__local const float* pi = &pixel[0][0][0] + addr2;       //输入图像
		__constant const float* ppw = pw;//卷积核
		
		__local const float* ppi = pi + tidy* BY * (BS3*BX+filtersize-1) + tidx * BX;//输入图像

		
        for(wy = 0; wy < kernel_height; wy++){
            for(wx = 0; wx < kernel_width; wx++){
					float filterItem = ppw[wy*kernel_width+wx];
					for(int i = 0;i < BX;++ i){
						for(int j = 0;j < BY;++ j){

							sum[i+j*BX] += filterItem * ppi[(wy+j)*(BS3*BX+filtersize-1)+wx+i];

						}
					}
            }
        }
	}
	
		for(int i = 0;i < BX;++ i){
			for(int j = 0;j < BY;++ j){

				sum[j*BX + i] += bias[channel];
        		out[index + tidx*(BX-1)+i + (tidy*(BY-1)+j)*out_width] = tanh((float)sum[j*BX + i]);
			}
		}
		*/
		
	channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
	//local mem
	int tidy=get_local_id(1);
    int tidx=get_local_id(2);
	float local pixel[num_map_S2_CNN][BS3+filtersize-1][BS3+filtersize-1];
	for (int i=0; i<in_num; i++){
        int addr2 = i*in_width*in_height;
        pixel[i][tidy][tidx]=in[addr2 + y*in_width + x];
		if(tidx < filtersize -1 ) pixel[i][tidy][tidx+BS3] = in[addr2+y*in_width+x+BS3];
        if(tidy < filtersize -1 ) pixel[i][tidy+BS3][tidx] = in[addr2+(y+BS3)*in_width+x];
		if(tidx < filtersize -1 &&tidy < filtersize-1) 
			   				pixel[i][tidy+BS3][tidx+BS3] = in[addr2+(y+BS3)*in_width+x+BS3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
	//
    int  index = (channel*out_height*out_width) + y*out_width + x;
	float sum = 0.0;
	int inc = 0;
	int wx = 0;
	int wy = 0;
	out[index] = 0.0;
	for (inc=0; inc<in_num; inc++) {
		if (!tbl[inc*16+channel]) continue;
        int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = (inc)*(BS3+filtersize-1)*(BS3+filtersize-1);
		__constant const float* pw = weight + addr1;   //卷积核
		__local const float* pi = &pixel[0][0][0] + addr2;       //输入图像
		sum = 0.0;
		__constant const float* ppw = pw;
		__local const float* ppi = pi + tidy * (BS3+filtersize-1) + tidx;
        for(wy = 0; wy < kernel_height; wy++)  {
			for(wx = 0; wx < kernel_width; wx++) {
                sum += *ppw++ * ppi[wy * (BS3+filtersize-1)+ wx];
		    }
	     }
	     out[index] += sum;
	}
	out[index] += bias[channel];
	out[index] = tanh((float)(out[index]));
	
}

__kernel void  kernel_forward_s4(__global float *in,
                      constant float  *weight,
                      constant float  *bias,
                      __global float  *out,
                      int channel,
                      int out_width,
                      int out_height,
                      int kernel_width,
					  int kernel_height,
					  int in_num,
					  int in_width,
                      int in_height
					  )
{
	channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
	//local mem
	int tidy=get_local_id(1);
    int tidx=get_local_id(2);
	float local pixel[num_map_C3_CNN][BS4<<1][BS4<<1];
	pixel[channel][tidy][tidx] = in[channel*in_width*in_height+y*in_width+x];
	pixel[channel][tidy+BS4][tidx] = in[channel*in_width*in_height+(y+BS4)*in_width+x];
	pixel[channel][tidy][tidx+BS4] = in[channel*in_width*in_height+y*in_width+x+BS4];
	pixel[channel][tidy+BS4][tidx+BS4] = in[channel*in_width*in_height+(y+BS4)*in_width+x+BS4];
	barrier(CLK_LOCAL_MEM_FENCE);
    //float scale_factor = 1.0 / (kernel_width * kernel_height);
    //int block = in_width * in_height * channel;
    int rows = tidy * kernel_width;
	int cols = tidx * kernel_height;
	int index = (channel*out_height*out_width) + y*out_width + x;

	out[index] = 0.0;
	for (int m = 0; m < kernel_width; m++) {
		for (int n = 0; n < kernel_height; n++) {
            out[index] += weight[channel] * pixel[channel][rows + m][ cols + n];
		}
	}
	out[index] *= 0.25;  //scale_factor;
	out[index] += bias[channel] ;
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_c5(__global float *in,
                      __global float  *weight,//constant memory is 64KB
                      constant float  *bias,
                      __global float  *out,
                      int channel,
                      int out_width,
                      int out_height,
                      int kernel_width,
					  int kernel_height,
					  int in_num,
					  int in_width,
                      int in_height
					  )
{
	channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
	//local mem
	int tidy=get_local_id(1);
    int tidx=get_local_id(2);
	float local pixel[num_map_S4_CNN][BS5+filtersize-1][BS5+filtersize-1];
	for (int i=0; i<in_num; i++){//16
        int addr2 = i*in_width*in_height;
        for(int j = 0;j < 5;++ j){
			for(int k = 0;k < 5;++ k){
				pixel[i][tidy+j][tidx+k] = in[addr2 + (y+j)*in_width + x + k];
			}
		}
	}
    barrier(CLK_LOCAL_MEM_FENCE);
	//
    int  index = (channel*out_height*out_width) + y*out_width + x;
	float sum = 0.0;
	int inc = 0;
	int wx = 0;
	int wy = 0;
	out[index] = 0.0;
	for (inc=0; inc<in_num; inc++) {
        int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = (inc)*(BS5+filtersize-1)*(BS5+filtersize-1);
		__global const float* pw = weight + addr1;   //卷积核
		__local const float* pi = &pixel[0][0][0] + addr2;       //输入图像
		sum = 0.0;
		__global const float* ppw = pw;
		__local const float* ppi = pi + tidy * (BS5+filtersize-1) + tidx;
        for(wy = 0; wy < kernel_height; wy++)  {
			for(wx = 0; wx < kernel_width; wx++) {
                sum += *ppw++ * ppi[wy * (BS5+filtersize-1) + wx];
		    }
	     }
	     out[index] += sum;
	}
	out[index] += bias[channel];
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_output(__global float *in,//neuron_C5[]
                      constant float  *weight,//weight_output[]
                      constant float  *bias,//bias_output[]
                      __global float  *out,//neuron_output[]
                      int out_num,		//num_neuron_output_CNN 
					  int in_num        //num_neuron_C5_CNN 
					  )
{
	int i = get_global_id(0);//0..9
	//local mem
	int tidx = get_local_id(0);//0..9
	//float local pixel[num_map_C5_CNN];//120
	float local tw[num_map_C5_CNN][num_map_output_CNN];//120 X 10
	//
	int index = i;
	float sum = 0.0;
	for (int c = 0; c < in_num; c++) {//[1,120] [120,10]
		//pixel[c*10 + tidx] = in[c*10 + i];
		tw[c][tidx] = weight[c * out_num + i];
		barrier(CLK_LOCAL_MEM_FENCE);
		sum += tw[c][tidx] * in[c];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	out[index] = sum;
	out[index] += bias[i];
	out[index] = tanh((float)(out[index]));	
}


//==================================================================//
//		三.以下是constant memory + local memory优化的forward()内核函数
//==================================================================//
 __kernel void  kernel_forward_c1(__global float *in,//data_single_image->data_input_train
                      __constant float  *weight,//weight_C1 卷积核
                      __constant float  *bias,//每个特征图有自己的bias_C1[len_bias_C1_CNN]
                      __global float  *out,//neuron_C1[] 
                      int channel,//num_map_C1_CNN 特征图的数量 6 
                      int out_width,//width_image_c1_CNN 28
                      int out_height,//height_image_c1_CNN 28
                      int kernel_width,//卷积核 5
					  int kernel_height,//卷积核 5
					  int in_num,//num_map_input_CNN 特征图的数量
					  int in_width,//width_image_input_CNN 32 
                      int in_height//height_image_input_CNN 32
					  )
{//每个神经元节点输出是并行的
	channel = get_global_id(0);//out number map
    int  y = get_global_id(1);
    int  x = get_global_id(2);

        int tidy=get_local_id(1);//[0,BY1)
        int tidx=get_local_id(2);//[0,BX1)

        float local pixel[num_map_input_CNN][BS1+filtersize-1][BS1+filtersize-1];


        for (int i=0; i<in_num; i++)
        {
                int addr2 = i*in_width*in_height;
                pixel[i][tidy][tidx]=in[addr2 + y*in_width + x];
				if(tidx < filtersize -1 ) pixel[i][tidy][tidx+BS1] = in[addr2+y*in_width+x+BS1];
               if(tidy < filtersize -1 ) pixel[i][tidy+BS1][tidx] = in[addr2+(y+BS1)*in_width+x];
			   if(tidx < filtersize -1 &&tidy < filtersize-1) 
			   				pixel[i][tidy+BS1][tidx+BS1] = in[addr2+(y+BS1)*in_width+x+BS1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
		
        int  index = (channel*out_height*out_width) + y*out_width + x;
        float sum = 0.0;
		int inc = 0;
		int wx = 0;
		int wy = 0;
		out[index] = 0.0;
        for (inc=0; inc<in_num; inc++)
        {
		int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = inc*(BS1+filtersize-1)*(BS1+filtersize-1);
		sum = 0.0;
		__constant const float* pw = weight + addr1;   //卷积核,默认是__private变量
		__local const float* pi = &pixel[0][0][0] + addr2;       //输入图像
		__constant const float* ppw = pw;//卷积核
		__local const float* ppi = pi + tidy * (BS1+filtersize-1) + tidx;//输入图像

		
        for(wy = 0; wy < kernel_height; wy++)
                {
                        for(wx = 0; wx < kernel_width; wx++)
                        {
                sum += *ppw++ *ppi[wy*(BS1+filtersize-1)+wx]  ;//pixel[inc][tidy+wy][tidx+wx];
                    }
            }
			out[index] += sum;
        
		}
        out[index] += bias[channel];
        out[index] = tanh((float)out[index]);
		
}
/*
__kernel void  kernel_forward_s2(__global float *in,//neuron_C1
					  constant float  *weight,//weight_S2[] 
                      constant float  *bias,//每个特征图有自己的bias_C2[len_bias_C2_CNN]
                      __global float  *out,//neuron_S2[] 
                      int channel,//num_map_C2_CNN 特征图的数量 6 
                      int out_width,//width_image_C2_CNN 14
                      int out_height,//height_image_C2_CNN 14
                      int kernel_width, //池化核 2
					  int kernel_height,//池化核 2
					  int in_num,//num_map_C1_CNN 特征图的数量 6
					  int in_width,//width_image_C1_CNN 28
                      int in_height//height_image_C1_CNN 28
					  )
{
	channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
	//local mem
	int tidy=get_local_id(1);
    int tidx=get_local_id(2);
	float local pixel[num_map_C1_CNN][BS2<<1][BS2<<1];
	pixel[channel][tidy][tidx] = in[channel*in_width*in_height+y*in_width+x];
	pixel[channel][tidy+BS2][tidx] = in[channel*in_width*in_height+(y+BS2)*in_width+x];
	pixel[channel][tidy][tidx+BS2] = in[channel*in_width*in_height+y*in_width+x+BS2];
	pixel[channel][tidy+BS2][tidx+BS2] = in[channel*in_width*in_height+(y+BS2)*in_width+x+BS2];
	barrier(CLK_LOCAL_MEM_FENCE);
	//
    //float scale_factor = 1.0 / (kernel_width * kernel_height);
    //int block =	 channel;
    int rows = tidy * kernel_width;
	int cols = tidx * kernel_height;
	int index = (channel*out_height*out_width) + y*out_width + x;

	out[index] = 0.0;
	for (int m = 0; m < kernel_width; m++) {
		for (int n = 0; n < kernel_height; n++) {
            out[index] += weight[channel] * pixel[channel][rows + m][cols + n];
		}
	}
	out[index] *= 0.25;  //scale_factor 池化层是2*2的 1/4;
	out[index] += bias[channel] ;
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_c3(__global float *in,//neuron_S2[]
                      constant float  *weight,//weight_C3[]
                      constant float  *bias,//bias_C3[]
                      __global float  *out,//neuron_C3[]
                      int channel,//num_map_C3_CNN 特征图的数量 16 
                      int out_width,//width_image_c3_CNN 10
                      int out_height,//height_image_c3_CNN 10
                      int kernel_width,//卷积核 5
					  int kernel_height,//卷积核 5
					  int in_num,//num_map_S2_CNN 特征图的数量6
					  int in_width,//width_image_S2_CNN 14 
                      int in_height,//height_image_S2_CNN 14
                      __global bool  *tbl //bool tbl[6][16]
					  )
{
	channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
	//local mem
	int tidy=get_local_id(1);
    int tidx=get_local_id(2);
	float local pixel[num_map_S2_CNN][BS3+filtersize-1][BS3+filtersize-1];
	for (int i=0; i<in_num; i++){
        int addr2 = i*in_width*in_height;
        pixel[i][tidy][tidx]=in[addr2 + y*in_width + x];
		if(tidx < filtersize -1 ) pixel[i][tidy][tidx+BS3] = in[addr2+y*in_width+x+BS3];
        if(tidy < filtersize -1 ) pixel[i][tidy+BS3][tidx] = in[addr2+(y+BS3)*in_width+x];
		if(tidx < filtersize -1 &&tidy < filtersize-1) 
			   				pixel[i][tidy+BS3][tidx+BS3] = in[addr2+(y+BS3)*in_width+x+BS3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
	//
    int  index = (channel*out_height*out_width) + y*out_width + x;
	float sum = 0.0;
	int inc = 0;
	int wx = 0;
	int wy = 0;
	out[index] = 0.0;
	for (inc=0; inc<in_num; inc++) {
		if (!tbl[inc*16+channel]) continue;
        int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = (inc)*(BS3+filtersize-1)*(BS3+filtersize-1);
		__constant const float* pw = weight + addr1;   //卷积核
		__local const float* pi = &pixel[0][0][0] + addr2;       //输入图像
		sum = 0.0;
		__constant const float* ppw = pw;
		__local const float* ppi = pi + tidy * (BS3+filtersize-1) + tidx;
        for(wy = 0; wy < kernel_height; wy++)  {
			for(wx = 0; wx < kernel_width; wx++) {
                sum += *ppw++ * ppi[wy * (BS3+filtersize-1)+ wx];
		    }
	     }
	     out[index] += sum;
	}
	out[index] += bias[channel];
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_s4(__global float *in,
                      constant float  *weight,
                      constant float  *bias,
                      __global float  *out,
                      int channel,
                      int out_width,
                      int out_height,
                      int kernel_width,
					  int kernel_height,
					  int in_num,
					  int in_width,
                      int in_height
					  )
{
	channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
	//local mem
	int tidy=get_local_id(1);
    int tidx=get_local_id(2);
	float local pixel[num_map_C3_CNN][BS4<<1][BS4<<1];
	pixel[channel][tidy][tidx] = in[channel*in_width*in_height+y*in_width+x];
	pixel[channel][tidy+BS4][tidx] = in[channel*in_width*in_height+(y+BS4)*in_width+x];
	pixel[channel][tidy][tidx+BS4] = in[channel*in_width*in_height+y*in_width+x+BS4];
	pixel[channel][tidy+BS4][tidx+BS4] = in[channel*in_width*in_height+(y+BS4)*in_width+x+BS4];
	barrier(CLK_LOCAL_MEM_FENCE);
    //float scale_factor = 1.0 / (kernel_width * kernel_height);
    //int block = in_width * in_height * channel;
    int rows = tidy * kernel_width;
	int cols = tidx * kernel_height;
	int index = (channel*out_height*out_width) + y*out_width + x;

	out[index] = 0.0;
	for (int m = 0; m < kernel_width; m++) {
		for (int n = 0; n < kernel_height; n++) {
            out[index] += weight[channel] * pixel[channel][rows + m][ cols + n];
		}
	}
	out[index] *= 0.25;  //scale_factor;
	out[index] += bias[channel] ;
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_c5(__global float *in,
                      __global float  *weight,//constant memory is 64KB
                      constant float  *bias,
                      __global float  *out,
                      int channel,
                      int out_width,
                      int out_height,
                      int kernel_width,
					  int kernel_height,
					  int in_num,
					  int in_width,
                      int in_height
					  )
{
	channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
	//local mem
	int tidy=get_local_id(1);
    int tidx=get_local_id(2);
	float local pixel[num_map_S4_CNN][BS5+filtersize-1][BS5+filtersize-1];
	for (int i=0; i<in_num; i++){//16
        int addr2 = i*in_width*in_height;
        for(int j = 0;j < 5;++ j){
			for(int k = 0;k < 5;++ k){
				pixel[i][tidy+j][tidx+k] = in[addr2 + (y+j)*in_width + x + k];
			}
		}
	}
    barrier(CLK_LOCAL_MEM_FENCE);
	//
    int  index = (channel*out_height*out_width) + y*out_width + x;
	float sum = 0.0;
	int inc = 0;
	int wx = 0;
	int wy = 0;
	out[index] = 0.0;
	for (inc=0; inc<in_num; inc++) {
        int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = (inc)*(BS5+filtersize-1)*(BS5+filtersize-1);
		__global const float* pw = weight + addr1;   //卷积核
		__local const float* pi = &pixel[0][0][0] + addr2;       //输入图像
		sum = 0.0;
		__global const float* ppw = pw;
		__local const float* ppi = pi + tidy * (BS5+filtersize-1) + tidx;
        for(wy = 0; wy < kernel_height; wy++)  {
			for(wx = 0; wx < kernel_width; wx++) {
                sum += *ppw++ * ppi[wy * (BS5+filtersize-1) + wx];
		    }
	     }
	     out[index] += sum;
	}
	out[index] += bias[channel];
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_output(__global float *in,//neuron_C5[]
                      constant float  *weight,//weight_output[]
                      constant float  *bias,//bias_output[]
                      __global float  *out,//neuron_output[]
                      int out_num,		//num_neuron_output_CNN 
					  int in_num        //num_neuron_C5_CNN 
					  )
{
	int i = get_global_id(0);//0..9
	//local mem
	int tidx = get_local_id(0);//0..9
	//float local pixel[num_map_C5_CNN];//120
	float local tw[num_map_C5_CNN][num_map_output_CNN];//120 X 10
	//
	int index = i;
	float sum = 0.0;
	for (int c = 0; c < in_num; c++) {//[1,120] [120,10]
		//pixel[c*10 + tidx] = in[c*10 + i];
		tw[c][tidx] = weight[c * out_num + i];
		barrier(CLK_LOCAL_MEM_FENCE);
		sum += tw[c][tidx] * in[c];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	out[index] = sum;
	out[index] += bias[i];
	out[index] = tanh((float)(out[index]));	
}

//==================================================================//
//		二.以下是const memory优化的forward()内核函数
/*==================================================================//
__kernel void  kernel_forward_c1(__global float *in,//data_single_image->data_input_train
                      __constant float  *weight,//weight_C1 卷积核
                      __constant float  *bias,//每个特征图有自己的bias_C1[len_bias_C1_CNN]
                      __global float  *out,//neuron_C1[] 
                      int channel,//num_map_C1_CNN 特征图的数量 6 
                      int out_width,//width_image_c1_CNN 28
                      int out_height,//height_image_c1_CNN 28
                      int kernel_width,//卷积核 5
					  int kernel_height,//卷积核 5
					  int in_num,//num_map_input_CNN 特征图的数量
					  int in_width,//width_image_input_CNN 32 
                      int in_height//height_image_input_CNN 32
					  )
{//每个神经元节点输出是并行的
	//int local pixel[][]
	channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
	//printf("Channel %d \n",get_global_size(0));//printf("%channel = %d",channel);printf("\n");
	//printf("y %d \n",get_global_size(1));//printf("\ny = %d",y);printf("\n");
	//printf("x %d \n",get_global_size(2));//printf("\nx = %d",x);printf("\n");
    int  index = (channel*out_height*out_width) + y*out_width + x;
	float sum = 0.0;
	int inc = 0;
	int wx = 0;
	int wy = 0;
	out[index] = 0.0;
	for (inc=0; inc<in_num; inc++) {
        int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = (inc)*in_width*in_height;
		__constant const float* pw = weight + addr1;   //卷积核,默认是__private变量
		__global const float* pi = in + addr2;       //输入图像
		sum = 0.0;
		__constant const float* ppw = pw;//卷积核
		__global const float* ppi = pi + y * in_width + x;//输入图像

		for(wy = 0; wy < kernel_height; wy++)  {
			for(wx = 0; wx < kernel_width; wx++) {
                sum += *ppw++ * ppi[wy * in_width + wx];
		    }
	     }
	     out[index] += sum;
		 //printf("(%d, %.3lf) ",index,out[index]);
	}
	out[index] += bias[channel];
	out[index] = tanh((float)(out[index]));
	//printf("(%d, %.3lf) ",index,out[index]);
	//for(int i = 0;i < 20;++ i) printf("%.3lf ",out[i]);printf("\n");
}

__kernel void  kernel_forward_s2(__global float *in,//neuron_C1
					  constant float  *weight,//weight_S2[] 
                      constant float  *bias,//每个特征图有自己的bias_C2[len_bias_C2_CNN]
                      __global float  *out,//neuron_S2[] 
                      int channel,//num_map_C2_CNN 特征图的数量 6 
                      int out_width,//width_image_C2_CNN 14
                      int out_height,//height_image_C2_CNN 14
                      int kernel_width, //池化核 2
					  int kernel_height,//池化核 2
					  int in_num,//num_map_C1_CNN 特征图的数量 6
					  int in_width,//width_image_C1_CNN 28
                      int in_height//height_image_C1_CNN 28
					  )
{
	channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
    //float scale_factor = 1.0 / (kernel_width * kernel_height);
    int block = in_width * in_height * channel;
    int rows = y * kernel_width;
	int cols = x * kernel_height;
	int index = (channel*out_height*out_width) + y*out_width + x;

	out[index] = 0.0;
	for (int m = 0; m < kernel_width; m++) {
		for (int n = 0; n < kernel_height; n++) {
            out[index] += weight[channel] * in[(rows + m) * in_width + cols + n + block];
		}
	}
	out[index] *= 0.25;  //scale_factor 池化层是2*2的 1/4;
	out[index] += bias[channel] ;
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_c3(__global float *in,//neuron_S2[]
                      constant float  *weight,//weight_C3[]
                      constant float  *bias,//bias_C3[]
                      __global float  *out,//neuron_C3[]
                      int channel,//num_map_C3_CNN 特征图的数量 16 
                      int out_width,//width_image_c3_CNN 10
                      int out_height,//height_image_c3_CNN 10
                      int kernel_width,//卷积核 5
					  int kernel_height,//卷积核 5
					  int in_num,//num_map_S2_CNN 特征图的数量
					  int in_width,//width_image_S2_CNN 14 
                      int in_height,//height_image_S2_CNN 14
                      __global bool  *tbl //bool tbl[6][16]
					  )
{
	channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
    int  index = (channel*out_height*out_width) + y*out_width + x;
	float sum = 0.0;
	int inc = 0;
	int wx = 0;
	int wy = 0;
	out[index] = 0.0;
	for (inc=0; inc<in_num; inc++) {
		if (!tbl[inc*16+channel]) continue;
        int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = (inc)*in_width*in_height;
		__constant const float* pw = weight + addr1;   //卷积核
		__global const float* pi = in + addr2;       //输入图像
		sum = 0.0;
		__constant const float* ppw = pw;
		__global const float* ppi = pi + y * in_width + x;
        for(wy = 0; wy < kernel_height; wy++)  {
			for(wx = 0; wx < kernel_width; wx++) {
                sum += *ppw++ * ppi[wy * in_width + wx];
		    }
	     }
	     out[index] += sum;
	}
	out[index] += bias[channel];
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_s4(__global float *in,
                      constant float  *weight,
                      constant float  *bias,
                      __global float  *out,
                      int channel,
                      int out_width,
                      int out_height,
                      int kernel_width,
					  int kernel_height,
					  int in_num,
					  int in_width,
                      int in_height
					  )
{
	channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
    //float scale_factor = 1.0 / (kernel_width * kernel_height);
    int block = in_width * in_height * channel;
    int rows = y * kernel_width;
	int cols = x * kernel_height;
	int index = (channel*out_height*out_width) + y*out_width + x;

	out[index] = 0.0;
	for (int m = 0; m < kernel_width; m++) {
		for (int n = 0; n < kernel_height; n++) {
            out[index] += weight[channel] * in[(rows + m) * in_width + cols + n + block];
		}
	}
	out[index] *= 0.25;  //scale_factor;
	out[index] += bias[channel] ;
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_c5(__global float *in,
                      __global float  *weight,//constant memory is 64KB
                      constant float  *bias,
                      __global float  *out,
                      int channel,
                      int out_width,
                      int out_height,
                      int kernel_width,
					  int kernel_height,
					  int in_num,
					  int in_width,
                      int in_height
					  )
{
	channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
    int  index = (channel*out_height*out_width) + y*out_width + x;
	float sum = 0.0;
	int inc = 0;
	int wx = 0;
	int wy = 0;
	out[index] = 0.0;
	for (inc=0; inc<in_num; inc++) {
        int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = (inc)*in_width*in_height;
		__global const float* pw = weight + addr1;   //卷积核
		__global const float* pi = in + addr2;       //输入图像
		sum = 0.0;
		__global const float* ppw = pw;
		__global const float* ppi = pi + y * in_width + x;
        for(wy = 0; wy < kernel_height; wy++)  {
			for(wx = 0; wx < kernel_width; wx++) {
                sum += *ppw++ * ppi[wy * in_width + wx];
		    }
	     }
	     out[index] += sum;
	}
	out[index] += bias[channel];
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_output(__global float *in,//neuron_C5[]
                      constant float  *weight,//weight_output[]
                      constant float  *bias,//bias_output[]
                      __global float  *out,//neuron_output[]
                      int out_num,		//num_neuron_output_CNN 
					  int in_num        //num_neuron_C5_CNN 
					  )
{
	int i = get_global_id(0);
	int index = i;
	out[index] = 0.0;
	for (int c = 0; c < in_num; c++) {
		out[index] += weight[c * out_num + i] * in[c];
	}
	out[index] += bias[i];
	out[index] = tanh((float)(out[index]));	
}
*/






//==================================================================//
//				一.以下是没有优化的forward()内核函数
/*==================================================================//
__kernel void  kernel_forward_c1(__global float *in,//data_single_image->data_input_train
                      __global float  *weight,//weight_C1 卷积核
                      __global float  *bias,//每个特征图有自己的bias_C1[len_bias_C1_CNN]
                      __global float  *out,//neuron_C1[] 
                      int channel,//num_map_C1_CNN 特征图的数量 6 
                      int out_width,//width_image_c1_CNN 28
                      int out_height,//height_image_c1_CNN 28
                      int kernel_width,//卷积核 5
					  int kernel_height,//卷积核 5
					  int in_num,//num_map_input_CNN 特征图的数量
					  int in_width,//width_image_input_CNN 32 
                      int in_height//height_image_input_CNN 32
					  )
{//每个神经元节点输出是并行的
	//int local pixel[][]
	channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
	//printf("Channel %d \n",get_global_size(0));//printf("%channel = %d",channel);printf("\n");
	//printf("y %d \n",get_global_size(1));//printf("\ny = %d",y);printf("\n");
	//printf("x %d \n",get_global_size(2));//printf("\nx = %d",x);printf("\n");
    int  index = (channel*out_height*out_width) + y*out_width + x;
	float sum = 0.0;
	int inc = 0;
	int wx = 0;
	int wy = 0;
	out[index] = 0.0;
	for (inc=0; inc<in_num; inc++) {
        int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = (inc)*in_width*in_height;
		__global const float* pw = weight + addr1;   //卷积核,默认是__private变量
		__global const float* pi = in + addr2;       //输入图像
		sum = 0.0;
		__global const float* ppw = pw;//卷积核
		__global const float* ppi = pi + y * in_width + x;//输入图像

		for(wy = 0; wy < kernel_height; wy++)  {
			for(wx = 0; wx < kernel_width; wx++) {
                sum += *ppw++ * ppi[wy * in_width + wx];
		    }
	     }
	     out[index] += sum;
		 //printf("(%d, %.3lf) ",index,out[index]);
	}
	out[index] += bias[channel];
	out[index] = tanh((float)(out[index]));
	//printf("(%d, %.3lf) ",index,out[index]);
	//for(int i = 0;i < 20;++ i) printf("%.3lf ",out[i]);printf("\n");
}

__kernel void  kernel_forward_s2(__global float *in,//neuron_C1
					  __global float  *weight,//weight_S2[] 
                      __global float  *bias,//每个特征图有自己的bias_C2[len_bias_C2_CNN]
                      __global float  *out,//neuron_S2[] 
                      int channel,//num_map_C2_CNN 特征图的数量 6 
                      int out_width,//width_image_C2_CNN 14
                      int out_height,//height_image_C2_CNN 14
                      int kernel_width, //池化核 2
					  int kernel_height,//池化核 2
					  int in_num,//num_map_C1_CNN 特征图的数量 6
					  int in_width,//width_image_C1_CNN 28
                      int in_height//height_image_C1_CNN 28
					  )
{
	channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
    //float scale_factor = 1.0 / (kernel_width * kernel_height);
    int block = in_width * in_height * channel;
    int rows = y * kernel_width;
	int cols = x * kernel_height;
	int index = (channel*out_height*out_width) + y*out_width + x;

	out[index] = 0.0;
	for (int m = 0; m < kernel_width; m++) {
		for (int n = 0; n < kernel_height; n++) {
            out[index] += weight[channel] * in[(rows + m) * in_width + cols + n + block];
		}
	}
	out[index] *= 0.25;  //scale_factor 池化层是2*2的 1/4;
	out[index] += bias[channel] ;
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_c3(__global float *in,//neuron_S2[]
                      __global float  *weight,//weight_C3[]
                      __global float  *bias,//bias_C3[]
                      __global float  *out,//neuron_C3[]
                      int channel,//num_map_C3_CNN 特征图的数量 16 
                      int out_width,//width_image_c3_CNN 10
                      int out_height,//height_image_c3_CNN 10
                      int kernel_width,//卷积核 5
					  int kernel_height,//卷积核 5
					  int in_num,//num_map_S2_CNN 特征图的数量
					  int in_width,//width_image_S2_CNN 14 
                      int in_height,//height_image_S2_CNN 14
                      __global bool  *tbl //bool tbl[6][16]
					  )
{
	channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
    int  index = (channel*out_height*out_width) + y*out_width + x;
	float sum = 0.0;
	int inc = 0;
	int wx = 0;
	int wy = 0;
	out[index] = 0.0;
	for (inc=0; inc<in_num; inc++) {
		if (!tbl[inc*16+channel]) continue;
        int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = (inc)*in_width*in_height;
		__global const float* pw = weight + addr1;   //卷积核
		__global const float* pi = in + addr2;       //输入图像
		sum = 0.0;
		__global const float* ppw = pw;
		__global const float* ppi = pi + y * in_width + x;
        for(wy = 0; wy < kernel_height; wy++)  {
			for(wx = 0; wx < kernel_width; wx++) {
                sum += *ppw++ * ppi[wy * in_width + wx];
		    }
	     }
	     out[index] += sum;
	}
	out[index] += bias[channel];
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_s4(__global float *in,
                      __global float  *weight,
                      __global float  *bias,
                      __global float  *out,
                      int channel,
                      int out_width,
                      int out_height,
                      int kernel_width,
					  int kernel_height,
					  int in_num,
					  int in_width,
                      int in_height
					  )
{
	channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
    //float scale_factor = 1.0 / (kernel_width * kernel_height);
    int block = in_width * in_height * channel;
    int rows = y * kernel_width;
	int cols = x * kernel_height;
	int index = (channel*out_height*out_width) + y*out_width + x;

	out[index] = 0.0;
	for (int m = 0; m < kernel_width; m++) {
		for (int n = 0; n < kernel_height; n++) {
            out[index] += weight[channel] * in[(rows + m) * in_width + cols + n + block];
		}
	}
	out[index] *= 0.25;  //scale_factor;
	out[index] += bias[channel] ;
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_c5(__global float *in,
                      __global float  *weight,
                      __global float  *bias,
                      __global float  *out,
                      int channel,
                      int out_width,
                      int out_height,
                      int kernel_width,
					  int kernel_height,
					  int in_num,
					  int in_width,
                      int in_height
					  )
{
	channel = get_global_id(0);
    int  y = get_global_id(1);
    int  x = get_global_id(2);
    int  index = (channel*out_height*out_width) + y*out_width + x;
	float sum = 0.0;
	int inc = 0;
	int wx = 0;
	int wy = 0;
	out[index] = 0.0;
	for (inc=0; inc<in_num; inc++) {
        int addr1 = (in_num * channel + inc) * kernel_height * kernel_width;
		int addr2 = (inc)*in_width*in_height;
		__global const float* pw = weight + addr1;   //卷积核
		__global const float* pi = in + addr2;       //输入图像
		sum = 0.0;
		__global const float* ppw = pw;
		__global const float* ppi = pi + y * in_width + x;
        for(wy = 0; wy < kernel_height; wy++)  {
			for(wx = 0; wx < kernel_width; wx++) {
                sum += *ppw++ * ppi[wy * in_width + wx];
		    }
	     }
	     out[index] += sum;
	}
	out[index] += bias[channel];
	out[index] = tanh((float)(out[index]));
}

__kernel void  kernel_forward_output(__global float *in,//neuron_C5[]
                      __global float  *weight,//weight_output[]
                      __global float  *bias,//bias_output[]
                      __global float  *out,//neuron_output[]
                      int out_num,		//num_neuron_output_CNN 
					  int in_num        //num_neuron_C5_CNN 
					  )
{
	int i = get_global_id(0);
	int index = i;
	out[index] = 0.0;
	for (int c = 0; c < in_num; c++) {
		out[index] += weight[c * out_num + i] * in[c];
	}
	out[index] += bias[i];
	out[index] = tanh((float)(out[index]));	
}
*/
