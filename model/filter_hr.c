#include <math.h>
#include <string.h>
#include "./include/k2c_include.h"
#include "./include/k2c_tensor_include.h"

void filter_hr(k2c_tensor *input_1_input, k2c_tensor *dense_2_output)
{

	float dense_output_array[5] = {0};
	k2c_tensor dense_output = {&dense_output_array[0], 1, 5, {5, 1, 1, 1, 1}};
	float dense_kernel_array[70] = {
		+1.32475480e-01f,
		-2.07969129e-01f,
		-6.28441647e-02f,
		-2.23078877e-01f,
		+4.67773914e-01f,
		+1.65286988e-01f,
		-1.19773403e-01f,
		-3.31767201e-02f,
		-2.13741750e-01f,
		+3.92813295e-01f,
		+1.88546658e-01f,
		+9.24906656e-02f,
		-4.34105098e-02f,
		-3.25261533e-01f,
		+3.30314070e-01f,
		+8.06150436e-02f,
		-3.06826085e-01f,
		-1.11830756e-01f,
		-2.43933335e-01f,
		+5.26623845e-01f,
		-4.30477485e-02f,
		-2.18617827e-01f,
		-2.79851645e-01f,
		-1.50043219e-01f,
		+4.91588354e-01f,
		-1.85484946e-01f,
		+8.42012107e-01f,
		-1.16314605e-01f,
		-9.40099806e-02f,
		-3.64203811e-01f,
		-3.51195097e-01f,
		-1.14956714e-01f,
		-2.11638376e-01f,
		+1.83922753e-01f,
		+6.37402162e-02f,
		-2.03109905e-01f,
		+5.57329655e-01f,
		-4.85371612e-02f,
		+1.20075010e-01f,
		-3.50884706e-01f,
		+1.11253059e+00f,
		+8.78382921e-02f,
		+1.75404894e+00f,
		-1.99888319e-01f,
		-6.78730682e-02f,
		+1.05380464e+00f,
		-3.28812897e-01f,
		+1.69213068e+00f,
		-1.01909250e-01f,
		+2.30438828e-01f,
		+1.12415779e+00f,
		-5.29852994e-02f,
		+1.72619343e+00f,
		-2.91128904e-01f,
		+1.40579164e-01f,
		+8.04500103e-01f,
		+3.62950027e-01f,
		+6.63389564e-01f,
		-1.57677853e+00f,
		+5.57882905e-01f,
		+7.38817155e-01f,
		+5.52606165e-01f,
		+7.65338540e-01f,
		-1.41045296e+00f,
		+2.38556400e-01f,
		+5.30926704e-01f,
		+3.08245897e-01f,
		+5.19681573e-01f,
		-1.47998834e+00f,
		+4.20226306e-01f,
	};
	k2c_tensor dense_kernel = {&dense_kernel_array[0], 2, 70, {14, 5, 1, 1, 1}};
	float dense_bias_array[5] = {
		-3.18272281e+00f,
		+3.22764540e+00f,
		+1.96194828e+00f,
		+2.39016509e+00f,
		+3.72706556e+00f,
	};
	k2c_tensor dense_bias = {&dense_bias_array[0], 1, 5, {5, 1, 1, 1, 1}};
	float dense_fwork[84] = {0};

	float dense_1_output_array[3] = {0};
	k2c_tensor dense_1_output = {&dense_1_output_array[0], 1, 3, {3, 1, 1, 1, 1}};
	float dense_1_kernel_array[15] = {
		-2.81447721e+00f,
		-1.18621242e+00f,
		-2.66759825e+00f,
		+5.22655189e-01f,
		-1.26474881e+00f,
		+1.63659644e+00f,
		+1.23717082e+00f,
		+1.06179047e+00f,
		+1.17732739e+00f,
		+1.52103090e+00f,
		+1.31279671e+00f,
		+1.85956526e+00f,
		+1.58452201e+00f,
		-1.34456158e+00f,
		+1.42684138e+00f,
	};
	k2c_tensor dense_1_kernel = {&dense_1_kernel_array[0], 2, 15, {5, 3, 1, 1, 1}};
	float dense_1_bias_array[3] = {
		+2.62100363e+00f,
		-4.62743968e-01f,
		+2.52842808e+00f,
	};
	k2c_tensor dense_1_bias = {&dense_1_bias_array[0], 1, 3, {3, 1, 1, 1, 1}};
	float dense_1_fwork[20] = {0};

	float dense_2_kernel_array[3] = {
		+2.74574351e+00f,
		-4.06549311e+00f,
		+2.16978478e+00f,
	};
	k2c_tensor dense_2_kernel = {&dense_2_kernel_array[0], 2, 3, {3, 1, 1, 1, 1}};
	float dense_2_bias_array[1] = {
		+1.83690488e+00f,
	};
	k2c_tensor dense_2_bias = {&dense_2_bias_array[0], 1, 1, {1, 1, 1, 1, 1}};
	float dense_2_fwork[6] = {0};

	k2c_dense(&dense_output, input_1_input, &dense_kernel,
			  &dense_bias, k2c_relu, dense_fwork);
	k2c_dense(&dense_1_output, &dense_output, &dense_1_kernel,
			  &dense_1_bias, k2c_relu, dense_1_fwork);
	k2c_dense(dense_2_output, &dense_1_output, &dense_2_kernel,
			  &dense_2_bias, k2c_linear, dense_2_fwork);
}

void filter_hr_initialize()
{
}

void filter_hr_terminate()
{
}
