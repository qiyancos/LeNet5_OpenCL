// 各层图像大小
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
#define eps_CNN				        1e-8

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

__kernel void back_output(__const __global float * neuron_output,
	__const __global float * data_single_label, 
	__global float * delta_bias_output)
{
	const int id = get_global_id(0); 
	float output = neuron_output[id];
	float label = data_single_label[id];
	float temp1 = output - label;
	float temp2 = 1 - output * output;
	temp1 = temp1 * temp2;
	delta_bias_output[id] = temp1;
}

// output部分；一维；可以考虑用CPU实现
// globalWorkSize = num_neuron_output_CNN; 10

__kernel void back_c5(__const __global float* delta_bias_output, 
	__const __global float* weight_output,
	__const __global float* neuron_C5,
	__global float* delta_bias_C5,
	__global float* delta_weight_output)
{
	const int gid = get_group_id(0);
	const int mid = gid % num_neuron_C5_CNN;
	const int lid = get_local_id(0);

	int offset = mid * num_neuron_output_CNN + lid;
	float neuron = neuron_C5[mid];
	__local float res[num_neuron_output_CNN];
	
	if(gid < num_neuron_C5_CNN){
		res[lid] = delta_bias_output[lid] * weight_output[offset];
		barrier(CLK_LOCAL_MEM_FENCE);
		if(lid % block_edge3 == 0){
			float temp = res[lid];
			for(int i = 1; i < block_edge3; i++)
				temp += res[lid + i];
			res[lid] = temp;
			barrier(CLK_LOCAL_MEM_FENCE);
			if(lid == 0){
				temp += res[block_edge3];
				delta_bias_C5[mid] = (1 - neuron * neuron) * temp;
			}
		}
	}
	else delta_weight_output[offset] = delta_bias_output[lid] * neuron;
}

// c5优化2；一维；权重残差和神经元残差并行优化；
// globalWorkSize = 2 * num_neuron_output_CNN * num_neuron_C5_CNN; 240 * 10
// localWorkSize = num_neuron_output_CNN; 10

__kernel void back_s4(__const global float* weight_C5,
	__const __global float* delta_bias_C5,
	__const __global float* neuron_S4, 
	__global float* delta_weight_C5,
	__global float* delta_neuron_S4)
{
	const int gid = get_group_id(0);
	const int mid_s4 = gid % num_map_S4_CNN;
	const int lid = get_group_id(1);
	const int mid_c5 = get_local_id(0);

	int offset_s4 = mid_s4 * kernel_size + lid;
	int offset_c5 = mid_c5 * num_map_S4_CNN * kernel_size + offset_s4;
	__local float res[num_neuron_C5_CNN];
	float neuron = neuron_S4[offset_s4];

	if(gid < num_map_S4_CNN){
		res[mid_c5] = weight_C5[offset_c5] * delta_bias_C5[mid_c5];
		barrier(CLK_LOCAL_MEM_FENCE);
		if(mid_c5 % 5 == 0){
			float temp = res[mid_c5];
			for(int i = 1; i < 5; i++)
				temp += res[mid_c5 + i];
			res[mid_c5] = temp;
			barrier(CLK_LOCAL_MEM_FENCE);
			if(mid_c5 % 20 == 0){
				for(int i = 5; i < 20; i += 5)
					temp += res[mid_c5 + i];
				res[mid_c5] = temp;
				barrier(CLK_LOCAL_MEM_FENCE);
				if(mid_c5 % 60 == 0){
					temp += res[mid_c5 + 20];
					res[mid_c5] = temp + res[mid_c5 + 40];
					barrier(CLK_LOCAL_MEM_FENCE);
					if(mid_c5 == 0){
						temp = res[mid_c5] + res[mid_c5 + 60];
						delta_neuron_S4[offset_s4] = temp * (1 - neuron * neuron);
					}
				}
			}
		}
	}
	else delta_weight_C5[offset_c5] = neuron * delta_bias_C5[mid_c5];
}

// s4部分；二维；权重残差和神经元残差并行优化；
// globalWorkSize(0) = num_map_S4_CNN * num_neuron_C5_CNN * 2; 16 * 120 * 2
// globalWorkSize(1) = kernel_size; 25
// localWorkSize(0) = num_neuron_C5_CNN; 120
// localWorkSize(1) = num_neuron_C5_CNN; 1

__kernel void back_c3(__const __global float* weight_S4,
	__const __global float* delta_neuron_S4, 
	__const __global float* neuron_C3,
	__global float* delta_neuron_C3,
	__global float* delta_weight_S4)
{
	const int mid = get_group_id(0);
	const int gid = get_group_id(1);
	const int idy_c3 = get_local_id(0);
	const int idx_c3 = get_local_id(1);
	int idy_s4 = idy_c3 / size_pooling_CNN;
	int idx_s4 = idx_c3 / size_pooling_CNN;
	
	//int cnt = idx_c3 % size_pooling_CNN + idy % size_pooling_CNN * size_pooling_CNN;
	int offset_c3l = width_image_C3_CNN * idy_c3 + idx_c3;
	int offset_c3g = width_image_C3_CNN * height_image_C3_CNN * mid + offset_c3l;
	int offset_s4 = width_image_S4_CNN * (mid * height_image_S4_CNN + idy_s4) + idx_s4;
	float neuron = neuron_C3[offset_c3g];
	__local float res[height_image_C3_CNN * width_image_C3_CNN];

	if(gid){
		float gap = weight_S4[mid] * delta_neuron_S4[offset_s4];
		delta_neuron_C3[offset_c3g] = gap * scale_factor * (1 - neuron * neuron);
	}
	else{
		res[offset_c3l] = delta_neuron_S4[offset_s4] * neuron;
		barrier(CLK_LOCAL_MEM_FENCE);
		if(idx_c3 % gap_c3 == 0){
			float temp = res[offset_c3l];
			for(int i = 1; i < gap_c3; i++)
				temp += res[offset_c3l + i];
			res[offset_c3l] = temp;
			barrier(CLK_LOCAL_MEM_FENCE);
			if(idx_c3 == 0){
				temp += res[offset_c3l + gap_c3];
				res[offset_c3l] = temp;
				barrier(CLK_LOCAL_MEM_FENCE);
				if(idy_c3 % gap_c3 == 0){
					for(int i = 1; i < gap_c3; i++)
						temp += res[offset_c3l + i * width_image_C3_CNN];
					res[offset_c3l] = temp;
					barrier(CLK_LOCAL_MEM_FENCE);
					if(idy_c3 == 0){
						temp += res[offset_c3l + gap_c3 * width_image_C3_CNN];
						delta_weight_S4[mid] = temp * scale_factor;
					}
				}
			}
		}
	}
}

// c3部分；二维；权重残差和神经元残差并行优化；叠加运算并行优化
// globalWorkSize(0) = num_map_C3_CNN * height_image_C3_CNN; 16 * 10
// globalWorkSize(1) = width_image_C3_CNN * 2; 10 * 2
// localWorkSize(0) = height_image_C3_CNN; 10
// localWorkSize(1) = width_image_C3_CNN; 10

__kernel void back1_s2(__const __global float* weight_C3,
	__const __global float* delta_neuron_C3,
	__const __global float* neuron_S2,
	__global float* delta_neuron_S2)
{
	const int tbl[6][10] = {{0, 4, 5, 6, 9, 10, 11, 12, 14, 15},
		{0, 1, 5, 6, 7, 10, 11, 12, 13, 15},
		{0, 1, 2, 6, 7, 8, 11, 13, 14, 15},
		{1, 2, 3, 6, 7, 8, 9, 12, 14, 15},
		{2, 3, 4, 7, 8, 9, 10, 12, 13, 15},
		{3, 4, 5, 8, 9, 10, 11, 13, 14, 15}};
	
	const int mid_s2 = get_group_id(0);
	const int bid = get_group_id(1);
	int bidy = bid / 2;
	int bidx = bid % 2;
	const int lidy = get_local_id(0);
	const int lidx = get_local_id(1);
	const int mid_c3l = lidy / block_edge;
	const int mid_c3g = tbl[mid_s2][mid_c3l];

	int idyl = lidy % block_edge;
	int idxl = lidx;

	int idyg = idyl + block_edge * bidy;
	int idxg = idxl + block_edge * bidx;
	
	int offset_s2l = idyg * width_image_S2_CNN + idxg;
	int offset_s2g = offset_s2l + mid_s2 * width_image_S2_CNN * height_image_S2_CNN;
	
	float neuron = neuron_S2[offset_s2g];
	__local float kernel1[10][width_kernel_conv_CNN][height_kernel_conv_CNN];
	__local float block[10][block_edge][block_edge];
	__local float src_c3[10][height_image_C3_CNN][width_image_C3_CNN];

	int base1 = (num_map_S2_CNN * mid_c3g + mid_s2) * kernel_size;
	int base2 = mid_c3g * width_image_C3_CNN * height_image_C3_CNN;
	
	if(idxl < width_kernel_conv_CNN && idyl < height_kernel_conv_CNN){
		int offset1 = idyl * width_kernel_conv_CNN + idxl + base1;
		kernel1[mid_c3l][idyl][idxl] = weight_C3[offset1];

		int offset2 = idyl * width_image_C3_CNN + idxl + base2;
		src_c3[mid_c3l][idyl][idxl] = delta_neuron_C3[offset2];
		
		int offset21 = offset2 + 5 * width_image_C3_CNN;
		src_c3[mid_c3l][idyl + 5][idxl] = delta_neuron_C3[offset21];
		
		offset21 = offset2 + 5;
		src_c3[mid_c3l][idyl][idxl + 5] = delta_neuron_C3[offset21];
		
		offset21 += 5 * width_image_C3_CNN;
		src_c3[mid_c3l][idyl + 5][idxl + 5] = delta_neuron_C3[offset21];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	// Step1: Load Kernel and Neuron_C3 src

	float temp = 0;
	for(int i = 0; i < height_kernel_conv_CNN; i++){
		int ii = idyg - i;
		for(int j = 0; j < width_kernel_conv_CNN; j++){
			int jj = idxg - j;
			if(ii < 0 || ii >= height_image_C3_CNN || jj < 0 || jj >= width_image_C3_CNN)
				continue;
			else temp += kernel1[mid_c3l][i][j] * src_c3[mid_c3l][ii][jj];
		}
	}
	block[mid_c3l][idyl][idxl] = temp;
	barrier(CLK_LOCAL_MEM_FENCE);
	// Step2: Caculate 7*7 Block
	
	if(mid_c3l % 5 == 0){
		float temp = block[mid_c3l][idyl][idxl];
		for(int i = 1; i < 5; i++)
			temp += block[mid_c3l + i][idyl][idxl];
		block[mid_c3l][idyl][idxl] = temp;
		barrier(CLK_LOCAL_MEM_FENCE);
		if(mid_c3l == 0){
			temp = block[mid_c3l][idyl][idxl] + block[mid_c3l + 5][idyl][idxl];
			temp *= (1 - neuron * neuron);
			delta_neuron_S2[offset_s2g] = temp;
		}
	}
	// Step3: Add 16 Results And Write Back
}

// s2部分1；二维；细粒度并行化设计；
// globalWorkSize(0) = num_map_S2_CNN * width_image_S2_CNN * 10; 6 * 7 * 10
// globalWorkSize(1) = 4 * height_image_S2_CNN / 2; 4 * 7
// localWorkSize(0) = width_image_S2_CNN * 10 / 2; 7 * 10
// localWorkSize(1) = height_image_S2_CNN; 7

__kernel void back2_s2(__const __global float* neuron_S2,
	__const __global float* delta_neuron_C3,
	__global float* delta_weight_C3)
{
	const int tbl[6][10] = {{0, 4, 5, 6, 9, 10, 11, 12, 14, 15},
		{0, 1, 5, 6, 7, 10, 11, 12, 13, 15},
		{0, 1, 2, 6, 7, 8, 11, 13, 14, 15},
		{1, 2, 3, 6, 7, 8, 9, 12, 14, 15},
		{2, 3, 4, 7, 8, 9, 10, 12, 13, 15},
		{3, 4, 5, 8, 9, 10, 11, 13, 14, 15}};
	
	const int mid_c3l = get_group_id(0);
	const int kid = get_group_id(1);
	const int kidy = kid / width_kernel_conv_CNN;
	const int kidx = kid % width_kernel_conv_CNN;
	int mid_s2 = get_local_id(0);
	const int idy = mid_s2 % width_image_C3_CNN;
	const int idx = get_local_id(1);
	mid_s2 = mid_s2 / width_image_C3_CNN;
	const int mid_c3g = tbl[mid_s2][mid_c3l];

	int offset_s2 = width_image_S2_CNN * (height_image_S2_CNN * mid_s2 + kidy + idy) + kidx + idx;
	int offset_c3l = idy * width_image_C3_CNN + idx;
	int offset_c3g = offset_c3l + mid_c3g * width_image_C3_CNN * height_image_C3_CNN;
	int offset_dst = kernel_size * (num_map_S2_CNN * mid_c3g + mid_s2) + kid;
	
	__local float mid[num_map_S2_CNN][height_image_C3_CNN * width_image_C3_CNN];	

	mid[mid_s2][offset_c3l] = neuron_S2[offset_s2] * delta_neuron_C3[offset_c3g];
	barrier(CLK_LOCAL_MEM_FENCE);
	// Step1: Mul
	
	int base = offset_c3l;
	if(idx % 5 == 0){
		float temp = mid[mid_s2][base];
		for(int i = 1; i < 5; i++)
			temp += mid[mid_s2][base + i];
		mid[mid_s2][base] = temp;
		barrier(CLK_LOCAL_MEM_FENCE);
		if(idx == 0){
			mid[mid_s2][base] = temp + mid[mid_s2][base + 5];
			barrier(CLK_LOCAL_MEM_FENCE);
			if(idy % 5 == 0){
				temp = mid[mid_s2][base];
				for(int i = 1; i < 5; i++)
					temp += mid[mid_s2][base + i * width_image_C3_CNN];
				mid[mid_s2][base] = temp;
				barrier(CLK_LOCAL_MEM_FENCE);
				if(idy == 0) {
					delta_weight_C3[offset_dst] = temp + mid[mid_s2][5 * width_image_C3_CNN];
				}
			}
		}
	}
	// Step2: Add and Write Back;
}

// s2部分2；二维；Local优化；细粒度并行化设计
// globalWorkSize(0) = num_map_S2_CNN * 10 * width_image_C3_CNN; 6 * 10 * 10
// globalWorkSize(1) = kernel_size * height_image_C3_CNN; 25 * 10 
// localWorkSize(0) = width_image_C3_CNN * num_map_S2_CNN; 10 * 6
// localWorkSize(1) = height_image_C3_CNN; 10

__kernel void back_c1(__const __global float* delta_neuron_S2,
	__const __global float* neuron_C1,
	__const __global float* weight_S2,
	__global float* delta_neuron_C1,
	__global float* delta_weight_S2)
{
	const int mid = get_group_id(0);
	const int gid = get_group_id(1);
	const int idy_c1 = get_local_id(0);
	const int idx_c1 = get_local_id(1);
	int idy_s2 = idy_c1 / size_pooling_CNN;
	int idx_s2 = idx_c1 / size_pooling_CNN;

	//int cnt = idx_c1 % size_pooling_CNN + idy_c1 % size_pooling_CNN * size_pooling_CNN;
	int offset_c1l = width_image_C1_CNN * idy_c1 + idx_c1;
	int offset_c1g = width_image_C1_CNN * height_image_C1_CNN * mid + offset_c1l;
	int offset_s2 = width_image_S2_CNN * (mid * height_image_S2_CNN + idy_s2) + idx_s2;
	
	float neuron = neuron_C1[offset_c1g];
	__local float res[height_image_C1_CNN * width_image_C1_CNN];
	
	if(gid){
		float gap = weight_S2[mid] * delta_neuron_S2[offset_s2];
		delta_neuron_C1[offset_c1g] = gap * scale_factor * (1 - neuron * neuron);
	}
	else{
		res[offset_c1l] = delta_neuron_S2[offset_s2] * neuron;
		barrier(CLK_LOCAL_MEM_FENCE);
		if(idx_c1 % gap_c1 == 0){
			float temp = res[offset_c1l];
			for(int i = 1; i < gap_c1; i++)
				temp += res[offset_c1l + i];
			res[offset_c1l] = temp;
			barrier(CLK_LOCAL_MEM_FENCE);
			if(idx_c1 == 0){
				for(int i = 1; i < 4; i++)
					temp += res[offset_c1l + i * gap_c1];
				res[offset_c1l] = temp;
				barrier(CLK_LOCAL_MEM_FENCE);
				if(idy_c1 % gap_c1 == 0){
					for(int i = 1; i < gap_c1; i++)
						temp += res[offset_c1l + i * width_image_C1_CNN];
					res[offset_c1l] = temp;
					barrier(CLK_LOCAL_MEM_FENCE);
					if(idy_c1 == 0){
						int gap = gap_c1 * width_image_C1_CNN;
						for(int i = 1; i < 4; i++)
							temp += res[offset_c1l + i * gap];
						delta_weight_S2[mid] = temp * scale_factor;
					}
				}
			}
		}
	}
}

// c1部分；二维；权重残差和神经元残差的并行优化；叠加运算并行优化
// globalWorkSize(0) = num_map_C1_CNN * height_image_C1_CNN * 2; 6 * 28
// globalWorkSize(1) = width_image_C1_CNN * 2; 28 * 2
// localWorkSize(0) = height_image_C1_CNN; 28
// localWorkSize(1) = width_image_C1_CNN; 28

__kernel void back1_input(__const __global float* weight_C1,
	__const __global float* delta_neuron_C1,
	__const __global float* data_single_image,
	__global float* delta_neuron_input)
{
	const int bidy = get_group_id(0);
	const int bidx = get_group_id(1);
	int idyl = get_local_id(0); 
	int mid = idyl / (block_edge2 >> 1);
	idyl = idyl % (block_edge2 >> 1);
	int idxl = get_local_id(1);

	int idyg = idyl + (block_edge2 >> 1) * bidy;
	int idxg = idxl + block_edge2 * bidx;
	int offset_input = idyg * width_image_input_CNN + idxg;

	__local float kernel1[num_map_C1_CNN][height_kernel_conv_CNN][width_kernel_conv_CNN];
	__local float src_c1[num_map_C1_CNN][height_image_C1_CNN][width_image_C1_CNN];
	__local float block[num_map_C1_CNN][(block_edge2 >> 1)][block_edge2];
	
	int offset = mid * kernel_size + idyl * width_kernel_conv_CNN + idxl;
	if(idyl < height_kernel_conv_CNN && idxl < width_kernel_conv_CNN)
			kernel1[mid][idyl][idxl] = weight_C1[offset];
	if(idyl < block_edge && idxl < (block_edge << 1)){
		offset = width_image_C1_CNN * (mid * height_image_C1_CNN + idyl) + idxl;
		for(int i = 0; i < 4; i++){
			int y = i * block_edge + idyl;
			int offset_c1g = offset + i * block_edge * width_image_C1_CNN; 
			src_c1[mid][y][idxl] = delta_neuron_C1[offset_c1g];
			src_c1[mid][y][idxl + (block_edge << 1)] = delta_neuron_C1[offset_c1g + (block_edge << 1)];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	// Step1: Load Kernel And Src_C1

	float temp = 0;
	for(int i = 0; i < height_kernel_conv_CNN; i++){
		int ii = idyg - i;
		for(int j = 0; j < width_kernel_conv_CNN; j++){ 
			int jj = idxg - j;
			if(ii < 0 || ii >= height_image_C1_CNN || jj < 0 || jj >= width_image_C1_CNN)
				continue;
			else temp += kernel1[mid][i][j] * src_c1[mid][ii][jj];
		}
	}
	block[mid][idyl][idxl] = temp;
	barrier(CLK_LOCAL_MEM_FENCE);
	// Step2: Caculate 8 * 16 Block
	
	float neuron = data_single_image[offset_input];
	if(mid % 2 == 0){
		block[mid][idyl][idxl] += block[mid + 1][idyl][idxl];
		barrier(CLK_LOCAL_MEM_FENCE);
		if(mid == 0){
			temp = block[0][idyl][idxl] + block[2][idyl][idxl];
			temp += block[4][idyl][idxl];
			delta_neuron_input[offset_input] = temp;
		}
	}
	// Step3: Add 6 Results And Write Back
}

// input部分1；二维；Local优化；叠加运算并行优化
// globalWorkSize(0) = 4 * (block_edge2 >> 1) * num_map_C1_CNN; 4 * 8 * 6
// globalWorkSize(1) = 2 * block_edge2; 2 * 16
// localWorkSize(0) = (block_edge2 >> 1) * num_map_C1_CNN; 8 * 6
// localWorkSize(1) = block_edge2; 16

__kernel void back2_input(__const __global float* data_single_image,
	__const __global float* delta_neuron_C1,
	__global float* delta_weight_C1)
{
	const int mid = get_group_id(0);
	const int kid = get_group_id(1);
	const int kidy = kid / width_kernel_conv_CNN;
	const int kidx = kid % width_kernel_conv_CNN;
	const int idy = get_local_id(0);
	const int idx = get_local_id(1);
	
	int offset_input = (kidy + idy) * width_image_input_CNN + kidx + idx;
	int offset_c1g = (mid * height_image_C1_CNN + idy) * width_image_C1_CNN + idx;
	int offset_c1w = (mid * height_kernel_conv_CNN + kidy)* width_kernel_conv_CNN + kidx;

	__local float res[num_map_C1_CNN][height_image_C1_CNN][width_image_C1_CNN];
	
	res[mid][idy][idx] = delta_neuron_C1[offset_c1g] * data_single_image[offset_input];
	barrier(CLK_LOCAL_MEM_FENCE);
	// Step2: Mul
	
	if(idx % block_edge == 0){
		float temp = res[mid][idy][idx];
		for(int i = 1; i < 7; i++)
			temp += res[mid][idy][idx + i];
		res[mid][idy][idx] = temp;
		barrier(CLK_LOCAL_MEM_FENCE);
		if(idx == 0){
			for(int i = 1; i < 4; i++)
				temp += res[mid][idy][idx + i * block_edge];
			res[mid][idy][0] = temp;
			barrier(CLK_LOCAL_MEM_FENCE);
			if(idy % block_edge == 0){
				for(int i = 1; i < block_edge; i++)
					temp += res[mid][idy + i][0];
				res[mid][idy][0] = temp;
				barrier(CLK_LOCAL_MEM_FENCE);
				if(idy == 0){
					for(int i = 1; i < 4; i++)
						temp += res[mid][idy + i * block_edge][0];
					delta_weight_C1[offset_c1w] = temp;
				}
			}
		}
	}
	// Step3: Add and Write Back
}

// input部分2；二维；Local优化；
// globalWorkSize(0) = num_map_C1_CNN * height_image_C1_CNN; 6 * 28
// globalWorkSize(1) = kernel_size * width_image_C1_CNN; 25 * 28 
// localWorkSize(0) = height_image_C1_CNN; 28
// localWorkSize(1) = width_image_C1_CNN; 28
