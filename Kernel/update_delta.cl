#define width_image_input_CNN		32 //归一化图像宽
#define height_image_input_CNN		32 //归一化图像高
#define width_image_C1_CNN          28
#define height_image_C1_CNN		    28
#define width_image_S2_CNN		    14
#define height_image_S2_CNN		    14
#define width_image_C3_CNN		    10
#define height_image_C3_CNN		    10
#define width_image_S4_CNN		    5
#define height_image_S4_CNN		    5
#define width_image_C5_CNN		    1
#define height_image_C5_CNN		    1
#define width_image_output_CNN		1
#define height_image_output_CNN		1

// 卷积核大小
#define width_kernel_conv_CNN		5 //卷积核大小
#define height_kernel_conv_CNN		5
#define width_kernel_pooling_CNN	2
#define height_kernel_pooling_CNN	2
#define size_pooling_CNN		    2

// 特征图数量   feature maps
#define num_map_input_CNN		1 //输入层map个数
#define num_map_C1_CNN			6 //C1层map个数
#define num_map_S2_CNN			6 //S2层map个数
#define num_map_C3_CNN			16 //C3层map个数
#define num_map_S4_CNN			16 //S4层map个数
#define num_map_C5_CNN			120 //C5层map个数
#define num_map_output_CNN		10 //输出层map个数

// MNIST
#define num_patterns_train_CNN		60000  //60000 //训练模式对数(总数)
#define num_patterns_test_CNN		10000   //10000 //测试模式对数(总数)

// Train
#define num_epochs_CNN			    100   //最大迭代次数
#define accuracy_rate_CNN		    0.985 //要求达到的准确率
#define learning_rate_CNN		    0.01  //学习率
#define eps_CNN						1e-8

#define rate2 (float2)(learning_rate_CNN, learning_rate_CNN)
#define rate8 (float8)(learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN)
#define rate16 (float16)(learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN, learning_rate_CNN,learning_rate_CNN)

#define eps2 (float2)(eps_CNN, eps_CNN)
#define eps8 (float8)(eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN)
#define eps16 (float16)(eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN, eps_CNN)

//
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

#define num_neuron_input_CNN     1024 //输入层神经元数，32*32=1024
#define num_neuron_C1_CNN        4704 //C1层神经元数，28*28*6=4704
#define num_neuron_S2_CNN		 1176 //S2层神经元数，14*14*6=1176
#define num_neuron_C3_CNN		 1600 //C3层神经元数，10*10*16=1600
#define num_neuron_S4_CNN		 400  //S4层神经元数，5*5*16=400
#define num_neuron_C5_CNN		 120  //C5层神经元数，1*120=120
#define num_neuron_output_CNN    10   //输出层神经元数，1*10=10

#define local_size_c5 (num_neuron_C5_CNN << 1)
#define block_edge3 (num_neuron_output_CNN >> 1)
#define kernel_size (width_kernel_conv_CNN * height_kernel_conv_CNN)
#define scale_factor (1.0/(width_kernel_pooling_CNN * height_kernel_pooling_CNN))
#define gap_c3 (width_image_C3_CNN >> 1)
#define block_edge (width_image_S2_CNN >> 1)
#define gap_c1 (width_image_C1_CNN >> 2)
#define block_edge2 (width_image_input_CNN >> 1)

#define update(delta, e_weight, weight, rate, eps){\
	e_weight += delta * delta;\
	weight -= rate * delta / (sqrt(e_weight) + eps);\
}

__kernel void deltabias(
	//__const __global float* delta_neuron_output,
	//__const __global float* delta_neuron_C5,
	__const __global float* delta_neuron_S4,
	__const __global float* delta_neuron_C3,
	__const __global float* delta_neuron_S2, 
	__const __global float* delta_neuron_C1,
	//__global float* delta_bias_output,
	//__global float* delta_bias_C5,
	__global float* delta_bias_S4,
	__global float* delta_bias_C3,
	__global float* delta_bias_S2,
	__global float* delta_bias_C1)
{
	const int gid = get_group_id(0);
	const int idy = get_local_id(0);
	const int idx = get_local_id(1);
	/*	
	if(gid == 0){
		if(idy == 0 && idx < 10)
			delta_bias_output[idx] = delta_neuron_output[idx];
	}
	// ouput
	else if(gid == 1){
		if(idy < 6 && idx < 20){
			int offset = idy * 20 + idx;
			delta_bias_C5[offset] = delta_neuron_C5[offset];
		}
	}
	// C5
	else 
	*/
	if(gid < 1){
		if(idy < 20 && idx < 20){
			int mid = idx / 5 + ((idy / 5) << 2);
			int lidx = idx % 5;
			int lidy = idy % 5;
			int offset = (mid * height_image_S4_CNN + lidy) * width_image_S4_CNN + lidx;

			__local float res[num_map_S4_CNN][height_image_S4_CNN][width_image_S4_CNN];
			res[mid][lidy][lidx] = delta_neuron_S4[offset];
			barrier(CLK_LOCAL_MEM_FENCE);

			if(lidx == 0){
				float temp  = res[mid][lidy][0];
				for(int i = 1; i < width_image_S4_CNN; i++)
					temp += res[mid][lidy][i];
				res[mid][lidy][0] = temp;
				barrier(CLK_LOCAL_MEM_FENCE);
				if(lidy == 0){
					for(int i = 1; i < width_image_S4_CNN; i++)
						temp += res[mid][i][0];
					delta_bias_S4[mid] = temp;
				}
			}
		}
	}
	// S4
	else if(gid < 5){
		if(idy < 20 && idx < 20){
			int lmid = ((idy / width_image_C3_CNN) << 1) + idx / width_image_C3_CNN;
			int mid = ((gid - 1) << 2) + lmid;
			int lidx = idx % width_image_C3_CNN;
			int lidy = idy % width_image_C3_CNN;
			int offset = (mid * height_image_C3_CNN + lidy) * width_image_C3_CNN + lidx;
			
			__local float res[4][height_image_C3_CNN][width_image_C3_CNN];
			res[lmid][lidy][lidx] = delta_neuron_C3[offset];
			barrier(CLK_LOCAL_MEM_FENCE);

			if(lidx % 5 == 0){
				float temp = res[lmid][lidy][lidx];

				for(int i = 1; i < 5; i++)
					temp += res[lmid][lidy][lidx + i];
				res[lmid][lidy][lidx] = temp;
				barrier(CLK_LOCAL_MEM_FENCE);
				if(lidx == 0){
					res[lmid][lidy][0] = temp + res[lmid][lidy][5];
					barrier(CLK_LOCAL_MEM_FENCE);
					if(lidy % 5 == 0){
						temp = res[lmid][lidy][0];
						for(int i = 1; i < 5; i++)
							temp += res[lmid][lidy + i][0];
						res[lmid][lidy][0] = temp;
						barrier(CLK_LOCAL_MEM_FENCE);
						if(lidy == 0) 
							delta_bias_C3[mid] = temp + res[lmid][5][0];
					}
				}
			}
		}
	}
	// C3
	else if(gid < 8){
		if(idy < 14){
			int lmid = idx / width_image_S2_CNN;
			int mid = lmid + ((gid - 5) << 1);
			int lidy = idy;
			int lidx = idx % width_image_S2_CNN;
			int offset = (mid * height_image_S2_CNN + lidy) * width_image_S2_CNN + lidx;

			__local float res[2][height_image_S2_CNN][width_image_S2_CNN];
			res[lmid][lidy][lidx] = delta_neuron_S2[offset];
			barrier(CLK_LOCAL_MEM_FENCE);

			if(lidx % block_edge == 0){
				float temp = res[lmid][lidy][lidx];
				for(int i = 1; i < block_edge; i++)
					temp += res[lmid][lidy][lidx + i];
				res[lmid][lidy][lidx] = temp;
				barrier(CLK_LOCAL_MEM_FENCE);
				if(lidx == 0){
					res[lmid][lidy][0] += res[lmid][lidy][block_edge];
					barrier(CLK_LOCAL_MEM_FENCE);
					if(lidy % block_edge == 0){
						temp = res[lmid][lidy][0];
						for(int i = 1; i < block_edge; i++)
							temp += res[lmid][lidy + i][0];
						res[lmid][lidy][0] = temp;
						barrier(CLK_LOCAL_MEM_FENCE);
						if(lidy == 0)
							delta_bias_S2[mid] = temp + res[lmid][block_edge][0];
					}
				}
			}
		}
	}
	// S2
	else if(gid < 14){
		int mid = gid - 8;
		int lidy = idy;
		int lidx = idx;
		int offset = (mid * height_image_C1_CNN + lidy) * width_image_C1_CNN + lidx;
		
		__local float res[height_image_C1_CNN][width_image_C1_CNN];
		res[lidy][lidx] = delta_neuron_C1[offset];
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(lidx % 7 == 0){
			float temp = res[lidy][lidx];
			for(int i = 1; i < 7; i++)
				temp += res[lidy][lidx + i];
			res[lidy][lidx] = temp;
			barrier(CLK_LOCAL_MEM_FENCE);
			if(lidx == 0){
				for(int  i = 1; i < 4; i++)
					temp += res[lidy][lidx + i * 7];
				res[lidy][0] = temp;
				barrier(CLK_LOCAL_MEM_FENCE);
				if(lidy % block_edge == 0){
					for(int i = 1; i < 7; i++)
						temp += res[lidy + i][0];
					res[lidy][0] = temp;
					barrier(CLK_LOCAL_MEM_FENCE);
					if(lidy == 0){
						for(int i = 1; i < 4; i++)
							temp += res[lidy + i * 7][0];
						delta_bias_C1[mid] = temp;
					}
				}
			}
		}
	}
	// C1
}

// globalWorkSize(0) = (1 + 4 + 3 + 6) * height_image_C1_CNN; 14 * 28
// globalWorkSize(1) = width_image_C1_CNN; 28
// localWorkSize(0) = height_image_C1_CNN; 28
// localWorkSize(1) = width_image_C1_CNN; 28

__kernel void update_wb(
	__const __global float2* delta_weight_C1,
	__const __global float2* delta_bias_C1,
	__const __global float2* delta_weight_S2,
	__const __global float2* delta_bias_S2,
	__const __global float16* delta_weight_C3,
	__const __global float16* delta_bias_C3,
	__const __global float16* delta_weight_S4,
	__const __global float16* delta_bias_S4,
	__const __global float16* delta_weight_C5,
	__const __global float8* delta_bias_C5,
	__const __global float16* delta_weight_output,
	__const __global float2* delta_bias_output,
	
	__global float2* E_weight_C1,
	__global float2* E_bias_C1,
	__global float2* E_weight_S2,
	__global float2* E_bias_S2,
	__global float16* E_weight_C3,
	__global float16* E_bias_C3,
	__global float16* E_weight_S4,
	__global float16* E_bias_S4,
	__global float16* E_weight_C5,
	__global float8* E_bias_C5,
	__global float16* E_weight_output,
	__global float2* E_bias_output,
	
	__global float2* weight_C1,
	__global float2* bias_C1,
	__global float2* weight_S2,
	__global float2* bias_S2,
	__global float16* weight_C3,
	__global float16* bias_C3,
	__global float16* weight_S4,
	__global float16* bias_S4,
	__global float16* weight_C5,
	__global float8* bias_C5,
	__global float16* weight_output,
	__global float2* bias_output)
{
	const int id = get_global_id(0);
	int lid;
	if(id < 75){
		lid = id;
		float2 delta = delta_weight_C1[lid];
		update(delta, E_weight_C1[lid], weight_C1[lid], rate2, eps2)
	}
	else if(id < 78){
		lid = id - 75;
		float2 delta = delta_bias_C1[lid];
		update(delta, E_bias_C1[lid], bias_C1[lid], rate2, eps2)
	}
	else if(id < 81){
		lid = id -78;
		float2 delta = delta_weight_S2[lid];
		update(delta, E_weight_S2[lid], weight_S2[lid], rate2, eps2)
	}
	else if(id < 84){
		lid = id - 81;
		float2 delta = delta_bias_S2[lid];
		update(delta, E_bias_S2[lid], bias_S2[lid], rate2, eps2)
	}
	else if(id < 234){
		lid = id - 84;
		float16 delta = delta_weight_C3[lid];
		update(delta, E_weight_C3[lid], weight_C3[lid], rate16, eps16)
	}
	else if(id < 235){
		float16 delta = delta_bias_C3[0];
		update(delta, E_bias_C3[0], bias_C3[0], rate16, eps16)
	}
	else if(id < 236){
		float16 delta = delta_weight_S4[0];
		update(delta, E_weight_S4[0], weight_S4[0], rate16, eps16)
	}
	else if(id < 237){
		float16 delta = delta_bias_S4[0];
		update(delta, E_bias_S4[0], bias_S4[0], rate16, eps16)
	}
	else if(id < 3237){
		lid = id - 237;
		float16 delta = delta_weight_C5[lid];
		update(delta, E_weight_C5[lid], weight_C5[lid], rate16, eps16)
	}
	else if(id < 3252){
		lid = id - 3237;
		float8 delta = delta_bias_C5[lid];
		update(delta, E_bias_C5[lid], bias_C5[lid], rate8, eps8)
	}
	else if(id < 3327){
		lid = id - 3252;
		float16 delta = delta_weight_output[lid];
		update(delta, E_weight_output[lid], weight_output[lid], rate16, eps16)
	}
	else{
		lid = id - 3327;
		float2 delta = delta_bias_output[lid];
		update(delta, E_bias_output[lid], bias_output[lid], rate2, eps2)
	}
}

// global_WorkSize(0) = len_weight_C1_CNN / 2 + len_bias_C1_CNN / 2 + 
//						len_weight_S2_CNN / 2 + len_bias_S2_CNN / 2 + 
//						len_weight_C3_CNN / 16 + len_bias_C3_CNN / 16 + 
//						len_weight_S4_CNN / 16 + len_bias_S4_CNN / 16 + 
//						len_weight_C5_CNN / 16 + len_bias_C5_CNN / 8 + 
//						len_weight_output_CNN / 16 + len_bias_output_CNN / 2; 
//						150 / 2 + 3 + 3 + 3 + 
//						2400 / 16 + 1 + 1 + 1 + 
//						48000 / 16 + 120 / 8 + 1200 / 16 + 5
