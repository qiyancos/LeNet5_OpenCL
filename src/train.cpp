#include "cnn.h"

using namespace std;

struct  timeval tsBegin1, tsEnd1,tsBegin2, tsEnd2, ToltsBegin, ToltsEnd;
long  t1Duration,t2Duration;

int CNN::get_index(int x, int y, int channel, int width, int height, int depth)
{
	assert(x >= 0 && x < width);
	assert(y >= 0 && y < height);
	assert(channel >= 0 && channel < depth);
	return (height * channel + y) * width + x;
}

bool CNN::train()
{
	std::cout << "training" << std::endl;
	int iter = 0;
	for (iter = 0; iter < num_epochs_CNN; iter++) {//最大epoch次数100,每用一次训练集train一次叫做一个epoch
		std::cout << "epoch: " << iter + 1<<endl;
		gettimeofday(&ToltsBegin, NULL);
		t1Duration = 0;
		t2Duration = 0;
		for (int i = 0; i < num_patterns_train_CNN; i++) {//num_pattern_train_CNN = 600000，bachsize=1,iterations=600000
			//if(i > 10000) break;
			//printf("Pattern %d:\n", i);	
			data_single_image = data_input_train + i * num_neuron_input_CNN;
			data_single_label = data_output_train + i * num_neuron_output_CNN;

			//load data_single_image & data_single_label
			status  = clEnqueueWriteBuffer(cmdQueue, clbuf_data_single_image, CL_FALSE,0,sizeof(float)*num_neuron_input_CNN,data_single_image,0,NULL,NULL);
    		status |= clEnqueueWriteBuffer(cmdQueue, clbuf_data_single_label, CL_FALSE,0,sizeof(float)*num_neuron_output_CNN,data_single_label,0,NULL,NULL);
			if(status) printf("write data_single_image/label to device buffer error\n"); 
			//load finished
			
			gettimeofday(&tsBegin1, NULL);
			Forward_C1();
			Forward_S2();
			Forward_C3();
			Forward_S4();
			Forward_C5();
			Forward_output();
			gettimeofday(&tsEnd1, NULL);

			t1Duration += 1000000L * (tsEnd1.tv_sec - tsBegin1.tv_sec) + (tsEnd1.tv_usec - tsBegin1.tv_usec);
			//获取每1000次迭代前向传播时间 ms
			if (i % 1000 == 0 && i > 0) {
				printf("%d iteartions: forward duration: %ld ms,", i, t1Duration/1000);
				t1Duration = 0;	
			}

			gettimeofday(&tsBegin2, NULL);
			Backward_output();
			Backward_C5();
			Backward_S4();
			Backward_C3();
			Backward_S2();
			Backward_C1();
			Backward_input();
			
			DeltaBias();//delta_bias_ using delta_neuron_
			UpdateWeights();//all layers weight_(),bias_() using delta_bias_ and delta_weight
			gettimeofday(&tsEnd2, NULL);
			
			t2Duration += 1000000L * (tsEnd2.tv_sec - tsBegin2.tv_sec) + (tsEnd2.tv_usec - tsBegin2.tv_usec);
			//获取每1000次迭代前向传播时间 ms
			if (i % 1000 == 0 && i > 0) {
				printf("backward duration: %ld ms\n",t2Duration/1000);
				t2Duration = 0;	
			}
		}
		//测试
		//clFinish(cmdQueue); 
		float accuracyRate = test();
		std::cout << "    accuray rate: " << accuracyRate << std::endl;
		if (accuracyRate > accuracy_rate_CNN) {//直到某个epoch后测试率达到期望的accurate_rate_CNN=98.5%
			//saveModelFile("cnn.model");
			//std::cout << "generate cnn model" << std::endl;
			break;
		}
		//saveModelFile("cnn.model");
		//std::cout << "generate cnn model" << std::endl;
		//当前的某个epoch 训练+测试的时间
		gettimeofday(&ToltsEnd, NULL);
		t1Duration = 1000000L * (ToltsEnd.tv_sec - ToltsBegin.tv_sec) + (ToltsEnd.tv_usec - ToltsBegin.tv_usec);
		printf(" *******  every epoch : %ld ms ^_^ \n", t1Duration/1000);
	}
	if (iter == num_epochs_CNN) {
		saveModelFile("cnn.model");
		std::cout << "generate cnn model" << std::endl;
	}
    return true;
}

void CNN::update_weights_bias(const float* delta, float* e_weight, float* weight, int len)
{
	for (int i = 0; i < len; i++) {
		e_weight[i] += delta[i] * delta[i];
		weight[i] -= learning_rate_CNN * delta[i] / (std::sqrt(e_weight[i]) + eps_CNN);
	}
}

bool CNN::UpdateWeights()
{
	/*
	float temp_Ewc1[len_weight_C1_CNN] = {0};
	float temp_Ebc1[len_bias_C1_CNN] = {0};
	float temp_wc1[len_weight_C1_CNN] = {0};
	float temp_bc1[len_bias_C1_CNN] = {0};
	
	memcpy(temp_Ewc1, E_weight_C1, sizeof(temp_Ewc1));
	memcpy(temp_Ebc1, E_bias_C1, sizeof(temp_Ebc1));
	memcpy(temp_wc1, weight_C1, sizeof(temp_wc1));
	memcpy(temp_bc1, bias_C1, sizeof(temp_bc1));

	update_weights_bias(delta_weight_C1, temp_Ewc1, temp_wc1, len_weight_C1_CNN);
	update_weights_bias(delta_bias_C1, temp_Ebc1, temp_bc1, len_bias_C1_CNN);

	float temp_Ews2[len_weight_S2_CNN] = {0};
	float temp_Ebs2[len_bias_S2_CNN] = {0};
	float temp_ws2[len_weight_S2_CNN] = {0};
	float temp_bs2[len_bias_S2_CNN] = {0};

	memcpy(temp_Ews2, E_weight_S2, sizeof(temp_Ews2));
	memcpy(temp_Ebs2, E_bias_S2, sizeof(temp_Ebs2));
	memcpy(temp_ws2, weight_S2, sizeof(temp_ws2));
	memcpy(temp_bs2, bias_S2, sizeof(temp_bs2));

	update_weights_bias(delta_weight_S2, temp_Ews2, temp_ws2, len_weight_S2_CNN);
	update_weights_bias(delta_bias_S2, temp_Ebs2, temp_bs2, len_bias_S2_CNN);

	float temp_Ewc3[len_weight_C3_CNN] = {0};
	float temp_Ebc3[len_bias_C3_CNN] = {0};
	float temp_wc3[len_weight_C3_CNN] = {0};
	float temp_bc3[len_bias_C3_CNN] = {0};

	memcpy(temp_Ewc3, E_weight_C3, sizeof(temp_Ewc3));
	memcpy(temp_Ebc3, E_bias_C3, sizeof(temp_Ebc3));
	memcpy(temp_wc3, weight_C3, sizeof(temp_wc3));
	memcpy(temp_bc3, bias_C3, sizeof(temp_bc3));

	update_weights_bias(delta_weight_C3, temp_Ewc3, temp_wc3, len_weight_C3_CNN);
	update_weights_bias(delta_bias_C3, temp_Ebc3, temp_bc3, len_bias_C3_CNN);

	float temp_Ews4[len_weight_S4_CNN] = {0};
	float temp_Ebs4[len_bias_S4_CNN] = {0};
	float temp_ws4[len_weight_S4_CNN] = {0};
	float temp_bs4[len_bias_S4_CNN] = {0};

	memcpy(temp_Ews4, E_weight_S4, sizeof(temp_Ews4));
	memcpy(temp_Ebs4, E_bias_S4, sizeof(temp_Ebs4));
	memcpy(temp_ws4, weight_S4, sizeof(temp_ws4));
	memcpy(temp_bs4, bias_S4, sizeof(temp_bs4));
	
	update_weights_bias(delta_weight_S4, temp_Ews4, temp_ws4, len_weight_S4_CNN);
	update_weights_bias(delta_bias_S4, temp_Ebs4, temp_bs4, len_bias_S4_CNN);

	float temp_Ewc5[len_weight_C5_CNN] = {0};
	float temp_Ebc5[len_bias_C5_CNN] = {0};
	float temp_wc5[len_weight_C5_CNN] = {0};
	float temp_bc5[len_bias_C5_CNN] = {0};

	memcpy(temp_Ewc5, E_weight_C5, sizeof(temp_Ewc5));
	memcpy(temp_Ebc5, E_bias_C5, sizeof(temp_Ebc5));
	memcpy(temp_wc5, weight_C5, sizeof(temp_wc5));
	memcpy(temp_bc5, bias_C5, sizeof(temp_bc5));
	
	update_weights_bias(delta_weight_C5, temp_Ewc5, temp_wc5, len_weight_C5_CNN);
	update_weights_bias(delta_bias_C5, temp_Ebc5, temp_bc5, len_bias_C5_CNN);

	float temp_Ewo[len_weight_output_CNN] = {0};
	float temp_Ebo[len_bias_output_CNN] = {0};
	float temp_wo[len_weight_output_CNN] = {0};
	float temp_bo[len_bias_output_CNN] = {0};

	memcpy(temp_Ewo, E_weight_output, sizeof(temp_Ewo));
	memcpy(temp_Ebo, E_bias_output, sizeof(temp_Ebo));
	memcpy(temp_wo, weight_output, sizeof(temp_wo));
	memcpy(temp_bo, bias_output, sizeof(temp_bo));

	update_weights_bias(delta_weight_output, temp_Ewo, temp_wo, len_weight_output_CNN);
	update_weights_bias(delta_bias_output, temp_Ebo, temp_bo, len_bias_output_CNN);
	*/
	
	///*
    
	//status  = clEnqueueWriteBuffer(cmdQueue, clbuf_delta_weight_output, CL_FALSE, 0, sizeof(float)*len_weight_output_CNN, delta_weight_output, 0, NULL, NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_delta_weight_C5, CL_FALSE, 0, sizeof(float)*len_weight_C5_CNN, delta_weight_C5, 0, NULL, NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_delta_weight_S4, CL_FALSE, 0, sizeof(float)*len_weight_S4_CNN, delta_weight_S4, 0, NULL, NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_delta_weight_C3, CL_FALSE, 0, sizeof(float)*len_weight_C3_CNN, delta_weight_C3, 0, NULL, NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_delta_weight_S2, CL_FALSE, 0, sizeof(float)*len_weight_S2_CNN, delta_weight_S2, 0, NULL, NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_delta_weight_C1, CL_FALSE, 0, sizeof(float)*len_weight_C1_CNN, delta_weight_C1, 0, NULL, NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_delta_bias_output, CL_FALSE, 0, sizeof(float)*len_bias_output_CNN, delta_bias_output, 0, NULL, NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_delta_bias_C5, CL_FALSE, 0, sizeof(float)*len_bias_C5_CNN, delta_bias_C5, 0, NULL, NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_delta_bias_S4, CL_FALSE, 0, sizeof(float)*len_bias_S4_CNN, delta_bias_S4, 0, NULL, NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_delta_bias_C3, CL_FALSE, 0, sizeof(float)*len_bias_C3_CNN, delta_bias_C3, 0, NULL, NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_delta_bias_S2, CL_FALSE, 0, sizeof(float)*len_bias_S2_CNN, delta_bias_S2, 0, NULL, NULL);
	//status |= clEnqueueWriteBuffer(cmdQueue, clbuf_delta_bias_C1, CL_FALSE, 0, sizeof(float)*len_bias_C1_CNN, delta_bias_C1, 0, NULL, NULL);
	/*
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_E_weight_output, CL_FALSE, 0, sizeof(float)*len_weight_output_CNN, E_weight_output, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_E_weight_C5, CL_FALSE, 0, sizeof(float)*len_weight_C5_CNN, E_weight_C5, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_E_weight_S4, CL_FALSE, 0, sizeof(float)*len_weight_S4_CNN, E_weight_S4, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_E_weight_C3, CL_FALSE, 0, sizeof(float)*len_weight_C3_CNN, E_weight_C3, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_E_weight_S2, CL_FALSE, 0, sizeof(float)*len_weight_S2_CNN, E_weight_S2, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_E_weight_C1, CL_FALSE, 0, sizeof(float)*len_weight_C1_CNN, E_weight_C1, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_E_bias_output, CL_FALSE, 0, sizeof(float)*len_bias_output_CNN, E_bias_output, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_E_bias_C5, CL_FALSE, 0, sizeof(float)*len_bias_C5_CNN, E_bias_C5, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_E_bias_S4, CL_FALSE, 0, sizeof(float)*len_bias_S4_CNN, E_bias_S4, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_E_bias_C3, CL_FALSE, 0, sizeof(float)*len_bias_C3_CNN, E_bias_C3, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_E_bias_S2, CL_FALSE, 0, sizeof(float)*len_bias_S2_CNN, E_bias_S2, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_E_bias_C1, CL_FALSE, 0, sizeof(float)*len_bias_C1_CNN, E_bias_C1, 0, NULL, NULL);
	*/
	/*
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_weight_output, CL_FALSE, 0, sizeof(float)*len_weight_output_CNN, weight_output, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_weight_C5, CL_FALSE, 0, sizeof(float)*len_weight_C5_CNN, weight_C5, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_weight_S4, CL_FALSE, 0, sizeof(float)*len_weight_S4_CNN, weight_S4, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_weight_C3, CL_FALSE, 0, sizeof(float)*len_weight_C3_CNN, weight_C3, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_weight_S2, CL_FALSE, 0, sizeof(float)*len_weight_S2_CNN, weight_S2, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_weight_C1, CL_FALSE, 0, sizeof(float)*len_weight_C1_CNN, weight_C1, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_bias_output, CL_FALSE, 0, sizeof(float)*len_bias_output_CNN, bias_output, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_bias_C5, CL_FALSE, 0, sizeof(float)*len_bias_C5_CNN, bias_C5, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_bias_S4, CL_FALSE, 0, sizeof(float)*len_bias_S4_CNN, bias_S4, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_bias_C3, CL_FALSE, 0, sizeof(float)*len_bias_C3_CNN, bias_C3, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_bias_S2, CL_FALSE, 0, sizeof(float)*len_bias_S2_CNN, bias_S2, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_bias_C1, CL_FALSE, 0, sizeof(float)*len_bias_C1_CNN, bias_C1, 0, NULL, NULL);
	*/
    
    
	size_t globalWorkSize[1];
	globalWorkSize[0] = len_weight_C1_CNN / 2 + len_bias_C1_CNN / 2 + \
						len_weight_S2_CNN / 2 + len_bias_S2_CNN / 2 + \
						len_weight_C3_CNN / 16 + len_bias_C3_CNN / 16 + \
						len_weight_S4_CNN / 16 + len_bias_S4_CNN / 16 + \
						len_weight_C5_CNN / 16 + len_bias_C5_CNN / 8 + \
						len_weight_output_CNN / 16 + len_bias_output_CNN / 2;

	status = clEnqueueNDRangeKernel(cmdQueue,
		kernel17,
		1,
		NULL,
		globalWorkSize,
		NULL,
		0,
		NULL,
		NULL);
    check_error(status);
    clFinish(cmdQueue);
    //*/
	///*
	//clEnqueueReadBuffer(cmdQueue,clbuf_bias_output,CL_TRUE,0,sizeof(float)*len_bias_output_CNN,bias_output,0,NULL,NULL);
	//clEnqueueReadBuffer(cmdQueue,clbuf_bias_C5,CL_TRUE,0,sizeof(float)*len_bias_C5_CNN,bias_C5,0,NULL,NULL);
	//clEnqueueReadBuffer(cmdQueue,clbuf_bias_S4,CL_TRUE,0,sizeof(float)*len_bias_S4_CNN,bias_S4,0,NULL,NULL);
	//clEnqueueReadBuffer(cmdQueue,clbuf_bias_C3,CL_TRUE,0,sizeof(float)*len_bias_C3_CNN,bias_C3,0,NULL,NULL);
	//clEnqueueReadBuffer(cmdQueue,clbuf_bias_S2,CL_TRUE,0,sizeof(float)*len_bias_S2_CNN,bias_S2,0,NULL,NULL);
	//clEnqueueReadBuffer(cmdQueue,clbuf_bias_C1,CL_TRUE,0,sizeof(float)*len_bias_C1_CNN,bias_C1,0,NULL,NULL);
	//clEnqueueReadBuffer(cmdQueue,clbuf_weight_output,CL_TRUE,0,sizeof(float)*len_weight_output_CNN,weight_output,0,NULL,NULL);
	//clEnqueueReadBuffer(cmdQueue,clbuf_weight_C5,CL_TRUE,0,sizeof(float)*len_weight_C5_CNN,weight_C5,0,NULL,NULL);
	//clEnqueueReadBuffer(cmdQueue,clbuf_weight_S4,CL_TRUE,0,sizeof(float)*len_weight_S4_CNN,weight_S4,0,NULL,NULL);
	//clEnqueueReadBuffer(cmdQueue,clbuf_weight_C3,CL_TRUE,0,sizeof(float)*len_weight_C3_CNN,weight_C3,0,NULL,NULL);
	//clEnqueueReadBuffer(cmdQueue,clbuf_weight_S2,CL_TRUE,0,sizeof(float)*len_weight_S2_CNN,weight_S2,0,NULL,NULL);
	//clEnqueueReadBuffer(cmdQueue,clbuf_weight_C1,CL_TRUE,0,sizeof(float)*len_weight_C1_CNN,weight_C1,0,NULL,NULL);
	//*/
	/*
	for(int i = 0; i < len_weight_C1_CNN; i++)
		if(temp_wc1[i] - weight_C1[i] > 0.000002){
			printf("Weight_C1\n");
			exit(0);
		}
	for(int i = 0; i < len_bias_C1_CNN; i++)
		if(temp_bc1[i] - bias_C1[i] > 0.000002){
			printf("Bias_C1, %d, %f, %f\n", i, temp_bc1[i], bias_C1[i]);
			exit(0);
		}
	for(int i = 0; i < len_weight_S2_CNN; i++)
		if(temp_ws2[i] - weight_S2[i] > 0.000002){
			printf("Weight_S2\n");
			exit(0);
		}
	for(int i = 0; i < len_bias_S2_CNN; i++)
		if(temp_bs2[i] - bias_S2[i] > 0.000002){
			printf("Bias_S2\n");
			exit(0);
		}
	for(int i = 0; i < len_weight_C3_CNN; i++)
		if(temp_wc3[i] - weight_C3[i] > 0.000002){
			printf("Weight_C3\n");
			exit(0);
		}
	for(int i = 0; i < len_bias_C3_CNN; i++)
		if(temp_bc3[i] - bias_C3[i] > 0.000002){
			printf("Bias_C3\n");
			exit(0);
		}
	for(int i = 0; i < len_weight_S4_CNN; i++)
		if(temp_ws4[i] - weight_S4[i] > 0.000002){
			printf("Weight_S4\n");
			exit(0);
		}
	for(int i = 0; i < len_bias_S4_CNN; i++)
		if(temp_bs4[i] - bias_S4[i] > 0.000002){
			printf("Bias_S4\n");
			exit(0);
		}
	for(int i = 0; i < len_weight_C5_CNN; i++)
		if(temp_wc5[i] - weight_C5[i] > 0.000002){
			printf("Weight_C5\n");
			exit(0);
		}
	for(int i = 0; i < len_bias_C5_CNN; i++)
		if(temp_bc5[i] - bias_C5[i] > 0.000002){
			printf("Bias_C5\n");
			exit(0);
		}
	for(int i = 0; i < len_weight_output_CNN; i++)
		if(temp_wo[i] - weight_output[i] > 0.000002){
			printf("Weight_output\n");
			exit(0);
		}
	for(int i = 0; i < len_bias_output_CNN; i++)
		if(temp_bo[i] - bias_output[i] > 0.000002){
			printf("Bias_output, %d, %f, %f\n", i, temp_bo[i], bias_output[i]);
			exit(0);
		}
	*/
	/*
	update_weights_bias(delta_weight_C1, E_weight_C1, weight_C1, len_weight_C1_CNN);
	update_weights_bias(delta_bias_C1, E_bias_C1, bias_C1, len_bias_C1_CNN);

	update_weights_bias(delta_weight_S2, E_weight_S2, weight_S2, len_weight_S2_CNN);
	update_weights_bias(delta_bias_S2, E_bias_S2, bias_S2, len_bias_S2_CNN);

	update_weights_bias(delta_weight_C3, E_weight_C3, weight_C3, len_weight_C3_CNN);
	update_weights_bias(delta_bias_C3, E_bias_C3, bias_C3, len_bias_C3_CNN);

	update_weights_bias(delta_weight_S4, E_weight_S4, weight_S4, len_weight_S4_CNN);
	update_weights_bias(delta_bias_S4, E_bias_S4, bias_S4, len_bias_S4_CNN);

	update_weights_bias(delta_weight_C5, E_weight_C5, weight_C5, len_weight_C5_CNN);
	update_weights_bias(delta_bias_C5, E_bias_C5, bias_C5, len_bias_C5_CNN);

	update_weights_bias(delta_weight_output, E_weight_output, weight_output, len_weight_output_CNN);
	update_weights_bias(delta_bias_output, E_bias_output, bias_output, len_bias_output_CNN);
	*/
	return true;
}

float CNN::test()
{
	int count_accuracy = 0;
	//10000个训练集
	for (int num = 0; num < num_patterns_test_CNN; num++) {
		data_single_image = data_input_test + num * num_neuron_input_CNN;
		data_single_label = data_output_test + num * num_neuron_output_CNN;
		
		//load data_single_image & data_single_label
		status  = clEnqueueWriteBuffer(cmdQueue, clbuf_data_single_image, CL_FALSE,0,sizeof(float)*num_neuron_input_CNN,data_single_image,0,NULL,NULL);
    	status |= clEnqueueWriteBuffer(cmdQueue, clbuf_data_single_label, CL_FALSE,0,sizeof(float)*num_neuron_output_CNN,data_single_label,0,NULL,NULL);
		if(status) printf("write data_single_image/label to device buffer error\n"); 
		//load finished
			
		Forward_C1();
		Forward_S2();
		Forward_C3();
		Forward_S4();
		Forward_C5();
		Forward_output();

		//load neuron_output from device
		clEnqueueReadBuffer(cmdQueue,clbuf_neuron_output,CL_TRUE,0,sizeof(float)*num_neuron_output_CNN,neuron_output,0,NULL,NULL);
		//load finished

		int pos_t = -1;
		int pos_y = -2;
		float max_value_t = -9999.0;
		float max_value_y = -9999.0;
		//从输出节点neuron_output[0..9]中找最大值pos_t|pos_y
		for (int i = 0; i < num_neuron_output_CNN; i++) {
			if (neuron_output[i] > max_value_y) {
				max_value_y = neuron_output[i];
				pos_y = i;
			}

			if (data_single_label[i] > max_value_t) {
				max_value_t = data_single_label[i];
				pos_t = i;
			}
		}
		//分类正确
		if (pos_y == pos_t) {
			++count_accuracy;
		}
		// Copper Sleep(1);
	}
	return (count_accuracy * 1.0 / num_patterns_test_CNN);
}

bool CNN::DeltaBias()
{
	/*
	float temp_s4[len_bias_S4_CNN] = {0};
	float temp_c3[len_bias_C3_CNN] = {0};
	float temp_s2[len_bias_S2_CNN] = {0};
	float temp_c1[len_bias_C1_CNN] = {0};

	//dB[i] += current_delta[i]
	//init_variable(delta_bias_output, 0.0, len_bias_output_CNN);
		
	init_variable(delta_bias_S4, 0.0, len_bias_S4_CNN);
    for (int c = 0; c < num_map_C3_CNN; c++) {
    	float diff = 0;
		for (int y = 0; y < height_image_S4_CNN; y++) {
			for (int x = 0; x < width_image_S4_CNN; x++) {
				int index_out = get_index(x, y, c, width_image_S4_CNN, height_image_S4_CNN, num_map_S4_CNN);
				diff += delta_neuron_S4[index_out];
			}
		}
		temp_s4[c] += diff;
	}

	// accumulate db
	init_variable(delta_bias_C3, 0.0, len_bias_C3_CNN);
	for (int outc = 0; outc < len_bias_C3_CNN; outc++) {
		int addr1 = get_index(0, 0, outc, width_image_C3_CNN, height_image_C3_CNN, num_map_C3_CNN);
		const float* delta = &delta_neuron_C3[0] + addr1;

		for (int y = 0; y < height_image_C3_CNN; y++) {
			for (int x = 0; x < width_image_C3_CNN; x++) {
				temp_c3[outc] += delta[y * width_image_C3_CNN + x];
			}
		}
	}

	init_variable(delta_bias_S2, 0.0, len_bias_S2_CNN);
    for (int c = 0; c < num_map_C1_CNN; c++) {
    	float diff = 0.0;
		for (int y = 0; y < height_image_S2_CNN; y++) {
			for (int x = 0; x < width_image_S2_CNN; x++) {
				int index_out = get_index(x, y, c, width_image_S2_CNN, height_image_S2_CNN, num_map_S2_CNN);
				diff += delta_neuron_S2[index_out];
			}
		}
		temp_s2[c] += diff;
	}

	// accumulate db
    init_variable(delta_bias_C1, 0.0, len_bias_C1_CNN);
	for (int outc = 0; outc < len_bias_C1_CNN; outc++) {
		int addr1 = get_index(0, 0, outc, width_image_C1_CNN, height_image_C1_CNN, num_map_C1_CNN);
		const float* delta = &delta_neuron_C1[0] + addr1;

		for (int y = 0; y < height_image_C1_CNN; y++) {
			for (int x = 0; x < width_image_C1_CNN; x++) {
				temp_c1[outc] += delta[y * width_image_C1_CNN + x];
			}
		}
	}
	*/
	///*    
    
	size_t globalWorkSize[2];
	size_t localWorkSize[2];

    globalWorkSize[0] = (size_t)14 * height_image_C1_CNN;
	globalWorkSize[1] = (size_t)width_image_C1_CNN;
	localWorkSize[0] = (size_t)height_image_C1_CNN;
	localWorkSize[1] = (size_t)width_image_C1_CNN;

	status = clEnqueueNDRangeKernel(cmdQueue,
		kernel16,
		2,
		NULL,
		globalWorkSize,
		localWorkSize,
		0,
		NULL,
		NULL);
    check_error(status);
    clFinish(cmdQueue);
	//*/
	/*
	clEnqueueReadBuffer(cmdQueue,clbuf_delta_bias_S4,CL_TRUE,0,sizeof(float)*len_bias_S4_CNN,delta_bias_S4,0,NULL,NULL);
	clEnqueueReadBuffer(cmdQueue,clbuf_delta_bias_C3,CL_TRUE,0,sizeof(float)*len_bias_C3_CNN,delta_bias_C3,0,NULL,NULL);
	clEnqueueReadBuffer(cmdQueue,clbuf_delta_bias_S2,CL_TRUE,0,sizeof(float)*len_bias_S2_CNN,delta_bias_S2,0,NULL,NULL);
	clEnqueueReadBuffer(cmdQueue,clbuf_delta_bias_C1,CL_TRUE,0,sizeof(float)*len_bias_C1_CNN,delta_bias_C1,0,NULL,NULL);
	*/
	/*
	for(int i = 0; i < len_bias_S4_CNN; i++){
		if(temp_s4[i] - delta_bias_S4[i] > 0.000002){
			printf("DBias_S4, %d, %f, %f, %f\n", i, temp_s4[i], delta_bias_S4[i], temp_s4[i] - delta_bias_S4[i]);
			exit(0);
		}
	}
	for(int i = 0; i < len_bias_C3_CNN; i++)
		if(temp_c3[i] - delta_bias_C3[i] > 0.000002){
			printf("DBias_C3, %d, %f, %f, %f\n", i, temp_s4[i], delta_bias_S4[i], temp_s4[i] - delta_bias_S4[i]);
			exit(0);
		}
	for(int i = 0; i < len_bias_S2_CNN; i++)
		if(temp_s2[i] - delta_bias_S2[i] > 0.000002){
			printf("DBias_S2, %d, %f, %f, %f\n", i, temp_s2[i], delta_bias_S2[i], temp_s2[i] - delta_bias_S2[i]);
			exit(0);
		}
	for(int i = 0; i < len_bias_C1_CNN; i++)
		if(temp_c1[i] - delta_bias_C1[i] > 0.000002){
			printf("DBias_C1, %d, %f, %f, %f\n", i, temp_c1[i], delta_bias_C1[i], temp_c1[i] - delta_bias_C1[i]);
			exit(0);
		}
	*/
	/*
	init_variable(delta_bias_S4, 0.0, len_bias_S4_CNN);
    for (int c = 0; c < num_map_C3_CNN; c++) {
    	float diff = 0;
		for (int y = 0; y < height_image_S4_CNN; y++) {
			for (int x = 0; x < width_image_S4_CNN; x++) {
				int index_out = get_index(x, y, c, width_image_S4_CNN, height_image_S4_CNN, num_map_S4_CNN);
				diff += delta_neuron_S4[index_out];
			}
		}
		delta_bias_S4[c] += diff;
	}

	// accumulate db
	init_variable(delta_bias_C3, 0.0, len_bias_C3_CNN);
	for (int outc = 0; outc < len_bias_C3_CNN; outc++) {
		int addr1 = get_index(0, 0, outc, width_image_C3_CNN, height_image_C3_CNN, num_map_C3_CNN);
		const float* delta = &delta_neuron_C3[0] + addr1;

		for (int y = 0; y < height_image_C3_CNN; y++) {
			for (int x = 0; x < width_image_C3_CNN; x++) {
				delta_bias_C3[outc] += delta[y * width_image_C3_CNN + x];
			}
		}
	}

	init_variable(delta_bias_S2, 0.0, len_bias_S2_CNN);
    for (int c = 0; c < num_map_C1_CNN; c++) {
    	float diff = 0.0;
		for (int y = 0; y < height_image_S2_CNN; y++) {
			for (int x = 0; x < width_image_S2_CNN; x++) {
				int index_out = get_index(x, y, c, width_image_S2_CNN, height_image_S2_CNN, num_map_S2_CNN);
				diff += delta_neuron_S2[index_out];
			}
		}
		delta_bias_S2[c] += diff;
	}

	// accumulate db
    init_variable(delta_bias_C1, 0.0, len_bias_C1_CNN);
	for (int outc = 0; outc < len_bias_C1_CNN; outc++) {
		int addr1 = get_index(0, 0, outc, width_image_C1_CNN, height_image_C1_CNN, num_map_C1_CNN);
		const float* delta = &delta_neuron_C1[0] + addr1;

		for (int y = 0; y < height_image_C1_CNN; y++) {
			for (int x = 0; x < width_image_C1_CNN; x++) {
				delta_bias_C1[outc] += delta[y * width_image_C1_CNN + x];
			}
		}
	}
	*/
	return true;
}
