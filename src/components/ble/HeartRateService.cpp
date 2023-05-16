#include "components/ble/HeartRateService.h"
#include "components/heartrate/HeartRateController.h"
#include "components/motion/MotionController.h"
#include "components/ble/NimbleController.h"
#include <nrf_log.h>
#include <cmath>

using namespace Pinetime::Controllers;

constexpr ble_uuid16_t HeartRateService::heartRateServiceUuid;
constexpr ble_uuid16_t HeartRateService::heartRateMeasurementUuid;

namespace {
  int HeartRateServiceCallback(uint16_t /*conn_handle*/, uint16_t attr_handle, struct ble_gatt_access_ctxt* ctxt, void* arg) {
    auto* heartRateService = static_cast<HeartRateService*>(arg);
    return heartRateService->OnHeartRateRequested(attr_handle, ctxt);
  }
}

// TODO Refactoring - remove dependency to SystemTask
HeartRateService::HeartRateService(NimbleController& nimble, Controllers::HeartRateController& heartRateController, Controllers::MotionController& motionController)
  : nimble {nimble},
    heartRateController {heartRateController},
    motionController {motionController},
    characteristicDefinition {{.uuid = &heartRateMeasurementUuid.u,
                               .access_cb = HeartRateServiceCallback,
                               .arg = this,
                               .flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY,
                               .val_handle = &heartRateMeasurementHandle},
                              {0}},
    serviceDefinition {
      {/* Device Information Service */
       .type = BLE_GATT_SVC_TYPE_PRIMARY,
       .uuid = &heartRateServiceUuid.u,
       .characteristics = characteristicDefinition},
      {0},
    } {
  // TODO refactor to prevent this loop dependency (service depends on controller and controller depends on service)
  heartRateController.SetService(this);
}

void HeartRateService::Init() {
  int res = 0;
  res = ble_gatts_count_cfg(serviceDefinition);
  ASSERT(res == 0);

  res = ble_gatts_add_svcs(serviceDefinition);
  ASSERT(res == 0);
}

int HeartRateService::OnHeartRateRequested(uint16_t attributeHandle, ble_gatt_access_ctxt* context) {
  if (attributeHandle == heartRateMeasurementHandle) {
    NRF_LOG_INFO("HEARTRATE : handle = %d", heartRateMeasurementHandle);
    uint8_t buffer[2] = {0, heartRateController.HeartRate()}; // [0] = flags, [1] = hr value

    int res = os_mbuf_append(context->om, buffer, 2);
    return (res == 0) ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;
  }
  return 0;
}

float HeartRateService::normalize(float x, float mean, float std_inv)
{
  return (x - mean) * std_inv;
}

void k2c_idx2sub(const uint16_t idx, uint16_t * sub, const uint16_t * shape, const uint16_t ndim) {

    uint16_t idx2 = idx;
    for (int i=ndim-1; i>=0; --i) {
        sub[i] = idx2%shape[i];
        idx2 /= shape[i];
    }
}

uint16_t k2c_sub2idx(const uint16_t * sub, const uint16_t * shape, const uint16_t ndim) {

    uint16_t idx = 0;
    uint16_t temp = 0;
    for (uint16_t i=0; i<ndim; ++i) {
        temp = sub[i];
        for (uint16_t j=ndim-1; j>i; --j) {
            temp *= shape[j];
        }
        idx += temp;
    }
    return idx;
}

void k2c_matmul(float * C, const float * A, const float * B, const uint16_t outrows,
                const uint16_t outcols, const uint16_t innerdim) {

    // make sure output is empty
    memset(C, 0, outrows*outcols*sizeof(C[0]));

    for (uint16_t i = 0 ; i < outrows; ++i) {
        const uint16_t outrowidx = i*outcols;
        const uint16_t inneridx = i*innerdim;
        for (uint16_t k = 0; k < innerdim; ++k) {
            for (uint16_t j = 0;  j < outcols; ++j) {
                C[outrowidx+j] += A[inneridx+k] * B[k*outcols+j];
            }
        }
    }
}

void k2c_bias_add(k2c_tensor* A, const k2c_tensor* b) {

    for (uint16_t i=0; i<A->numel; i+=b->numel) {
        for (uint16_t j=0; j<b->numel; ++j) {
            A->array[i+j] += b->array[j];
        }
    }
}


void k2c_affine_matmul(float * C, const float * A, const float * B, const float * d,
                       const uint16_t outrows,const uint16_t outcols, const uint16_t innerdim) {

    // make sure output is empty
    memset(C, 0, outrows*outcols*sizeof(C[0]));

    for (uint16_t i = 0 ; i < outrows; ++i) {
        const uint16_t outrowidx = i*outcols;
        const uint16_t inneridx = i*innerdim;
        for (uint16_t j = 0;  j < outcols; ++j) {
            for (uint16_t k = 0; k < innerdim; ++k) {
                C[outrowidx+j] += A[inneridx+k] * B[k*outcols+j];
            }
            C[outrowidx+j] += d[j];
        }
    }
}

void k2c_dot(k2c_tensor* C, const k2c_tensor* A, const k2c_tensor* B, const uint16_t * axesA,
             const uint16_t * axesB, const uint16_t naxes, const int normalize, float * fwork) {

    uint16_t permA[K2C_MAX_NDIM];
    uint16_t permB[K2C_MAX_NDIM];
    uint16_t prod_axesA = 1;
    uint16_t prod_axesB = 1;
    uint16_t free_axesA, free_axesB;
    uint16_t freeA[K2C_MAX_NDIM];
    uint16_t freeB[K2C_MAX_NDIM];
    uint16_t count;
    int isin;
    uint16_t newshpA[K2C_MAX_NDIM];
    uint16_t newshpB[K2C_MAX_NDIM];
    const uint16_t ndimA = A->ndim;
    const uint16_t ndimB = B->ndim;
    float *reshapeA = &fwork[0];   // temp working storage
    float *reshapeB = &fwork[A->numel];
    uint16_t Asub[K2C_MAX_NDIM];
    uint16_t Bsub[K2C_MAX_NDIM];
    // find which axes are free (ie, not being summed over)
    count=0;
    for (uint16_t i=0; i<ndimA; ++i) {
        isin = 0;
        for (uint16_t j=0; j<naxes; ++j) {
            if (i==axesA[j]) {
                isin=1;
            }
        }
        if (!isin) {
            freeA[count] = i;
            ++count;
        }
    }
    count=0;
    for (uint16_t i=0; i<ndimB; ++i) {
        isin = 0;
        for (uint16_t j=0; j<naxes; ++j) {
            if (i==axesB[j]) {
                isin=1;
            }
        }
        if (!isin) {
            freeB[count] = i;
            ++count;
        }
    }

    // number of elements in inner dimension
    for (uint16_t i=0; i < naxes; ++i) {
        prod_axesA *= A->shape[axesA[i]];
    }
    for (uint16_t i=0; i < naxes; ++i) {
        prod_axesB *= B->shape[axesB[i]];
    }
    // number of elements in free dimension
    free_axesA = A->numel/prod_axesA;
    free_axesB = B->numel/prod_axesB;
    // find permutation of axes to get into matmul shape
    for (uint16_t i=0; i<ndimA-naxes; ++i) {
        permA[i] = freeA[i];
    }
    for (uint16_t i=ndimA-naxes, j=0; i<ndimA; ++i, ++j) {
        permA[i] = axesA[j];
    }
    for (uint16_t i=0; i<naxes; ++i) {
        permB[i] = axesB[i];
    }
    for (uint16_t i=naxes, j=0; i<ndimB; ++i, ++j) {
        permB[i] = freeB[j];
    }



    for (uint16_t i=0; i<ndimA; ++i) {
        newshpA[i] = A->shape[permA[i]];
    }
    for (uint16_t i=0; i<ndimB; ++i) {
        newshpB[i] = B->shape[permB[i]];
    }

    // reshape arrays
    for (uint16_t i=0; i<A->numel; ++i) {
        k2c_idx2sub(i,Asub,A->shape,ndimA);
        for (uint16_t j=0; j<ndimA; ++j) {
            Bsub[j] = Asub[permA[j]];
        }
        uint16_t bidx = k2c_sub2idx(Bsub,newshpA,ndimA);
        reshapeA[bidx] = A->array[i];
    }

    for (uint16_t i=0; i<B->numel; ++i) {
        k2c_idx2sub(i,Bsub,B->shape,ndimB);
        for (uint16_t j=0; j<ndimB; ++j) {
            Asub[j] = Bsub[permB[j]];
        }
        uint16_t bidx = k2c_sub2idx(Asub,newshpB,ndimB);
        reshapeB[bidx] = B->array[i];
    }


    if (normalize) {

        float sum;
        float inorm;
        for (uint16_t i=0; i<free_axesA; ++i) {
            sum = 0;
            for (uint16_t j=0; j<prod_axesA; ++j) {
                sum += reshapeA[i*prod_axesA + j]*reshapeA[i*prod_axesA + j];
            }
            inorm = 1.0f/sqrtf(sum);
            for (uint16_t j=0; j<prod_axesA; ++j) {
                reshapeA[i*prod_axesA + j] *= inorm;
            }
        }
        for (uint16_t i=0; i<free_axesB; ++i) {
            sum = 0;
            for (uint16_t j=0; j<prod_axesB; ++j) {
                sum += reshapeB[i + free_axesB*j]*reshapeB[i + free_axesB*j];
            }
            inorm = 1.0f/sqrtf(sum);
            for (uint16_t j=0; j<prod_axesB; ++j) {
                reshapeB[i + free_axesB*j] *= inorm;
            }
        }
    }

    k2c_matmul(C->array, reshapeA, reshapeB, free_axesA,
               free_axesB, prod_axesA);
}

void k2c_linear_func(float * x, const uint16_t size) {
  (void) x;
  (void) size;
}
k2c_activationType * k2c_linear = k2c_linear_func;


void k2c_relu_func(float *x, const uint16_t size){
  for (uint16_t i=0; i < size; ++i) {
    if (x[i] <= 0.0f) {
        x[i] = 0.0f;
    }
  }
}
k2c_activationType * k2c_relu = k2c_relu_func;

void k2c_dense(k2c_tensor* output, const k2c_tensor* input, const k2c_tensor* kernel,
               const k2c_tensor* bias, k2c_activationType *activation, float * fwork) {

    if (input->ndim <=2) {
        uint16_t outrows;

        if (input->ndim>1) {
            outrows = input->shape[0];
        }
        else {
            outrows = 1;
        }
        const uint16_t outcols = kernel->shape[1];
        const uint16_t innerdim = kernel->shape[0];
        const uint16_t outsize = outrows*outcols;
        k2c_affine_matmul(output->array,input->array,kernel->array,bias->array,
                          outrows,outcols,innerdim);
        activation(output->array,outsize);
    }
    else {
        const uint16_t axesA[1] = {uint16_t(input->ndim-1)};
        const uint16_t axesB[1] = {0};
        const uint16_t naxes = 1;
        const int normalize = 0;

        k2c_dot(output, input, kernel, axesA, axesB, naxes, normalize, fwork);
        k2c_bias_add(output, bias);
        activation(output->array, output->numel);
    }
}

void filter_hr(k2c_tensor *input_1_input, k2c_tensor *dense_2_output)
{

	float dense_output_array[6] = {0};
	k2c_tensor dense_output = {&dense_output_array[0], 1, 6, {6, 1, 1, 1, 1}};
	float dense_kernel_array[96] = {
		-2.58158654e-01f,
		-2.44613945e-01f,
		+1.08642466e-01f,
		+4.85975236e-01f,
		-6.88496903e-02f,
		-7.39336386e-02f,
		-3.37832943e-02f,
		-9.29711834e-02f,
		+6.06957190e-02f,
		+1.55360013e-01f,
		-9.12086889e-02f,
		+1.50157744e-02f,
		+1.10843286e-01f,
		+1.67677894e-01f,
		+5.93186431e-02f,
		+2.50057787e-01f,
		-7.85016194e-02f,
		-2.44503081e-01f,
		-1.96207389e-01f,
		-2.31089577e-01f,
		+1.37382984e-01f,
		+1.68632016e-01f,
		-7.93497190e-02f,
		+1.93708673e-01f,
		-1.61212757e-02f,
		-5.61313592e-02f,
		+5.87342270e-02f,
		+3.39311928e-01f,
		-1.13524877e-01f,
		-1.58384189e-01f,
		-1.09716266e-01f,
		-2.35723734e-01f,
		+1.32731926e-02f,
		+3.01779091e-01f,
		-6.66517690e-02f,
		-2.28142608e-02f,
		-2.25188568e-01f,
		-3.80537182e-01f,
		+8.05058051e-03f,
		+5.03761232e-01f,
		-1.84331432e-01f,
		-4.32336815e-02f,
		-6.72802687e-01f,
		+4.54484582e-01f,
		+1.66620463e-02f,
		-8.81252736e-02f,
		-6.27342343e-01f,
		+7.84739628e-02f,
		-8.71884882e-01f,
		+2.76478410e-01f,
		-2.51699123e-04f,
		+5.04019186e-02f,
		-4.74103421e-01f,
		+4.30929400e-02f,
		-6.13183916e-01f,
		+3.92687201e-01f,
		+2.43710782e-02f,
		+5.74506037e-02f,
		-6.20829701e-01f,
		-4.39548343e-02f,
		-7.51653492e-01f,
		+1.28839898e+00f,
		+1.02290559e+00f,
		-1.69256870e-02f,
		-3.13063353e-01f,
		+1.95472986e-01f,
		-8.97551715e-01f,
		+1.01045191e+00f,
		+1.01934123e+00f,
		+2.34510422e-01f,
		-2.55330175e-01f,
		+2.16395304e-01f,
		-7.66702175e-01f,
		+1.07088792e+00f,
		+1.00034547e+00f,
		+1.23643830e-01f,
		-1.42470032e-01f,
		+2.78010130e-01f,
		-7.73911059e-01f,
		+5.99939585e-01f,
		+4.36002851e-01f,
		+1.53453469e-01f,
		-5.13873875e-01f,
		+7.38157406e-02f,
		-8.00532341e-01f,
		+8.04396689e-01f,
		+4.71778840e-01f,
		-6.46034181e-02f,
		-3.08885366e-01f,
		+1.83434457e-01f,
		-1.24854648e+00f,
		+2.55470514e-01f,
		+3.40454400e-01f,
		-4.13677335e-01f,
		-5.71978688e-01f,
		+7.69954324e-01f,
	};
	k2c_tensor dense_kernel = {&dense_kernel_array[0], 2, 96, {16, 6, 1, 1, 1}};
	float dense_bias_array[6] = {
		+1.86099494e+00f,
		+2.36612797e+00f,
		-2.70977783e+00f,
		+2.99464345e+00f,
		-2.20068145e+00f,
		+2.46976781e+00f,
	};
	k2c_tensor dense_bias = {&dense_bias_array[0], 1, 6, {6, 1, 1, 1, 1}};
	float dense_fwork[112] = {0};

	float dense_1_output_array[3] = {0};
	k2c_tensor dense_1_output = {&dense_1_output_array[0], 1, 3, {3, 1, 1, 1, 1}};
	float dense_1_kernel_array[18] = {
		+1.43555534e+00f,
		+1.09452999e+00f,
		+5.86435981e-02f,
		+8.85292470e-01f,
		+2.20836803e-01f,
		+1.43920517e+00f,
		-1.90961528e+00f,
		-2.27426362e+00f,
		-1.75510037e+00f,
		+2.05552316e+00f,
		+5.04008412e-01f,
		+7.35774994e-01f,
		-1.65641284e+00f,
		-1.49038696e+00f,
		-2.19002080e+00f,
		+1.61937487e+00f,
		+1.65267706e+00f,
		+6.40851200e-01f,
	};
	k2c_tensor dense_1_kernel = {&dense_1_kernel_array[0], 2, 18, {6, 3, 1, 1, 1}};
	float dense_1_bias_array[3] = {
		+1.97888076e+00f,
		+2.54777026e+00f,
		+2.63073134e+00f,
	};
	k2c_tensor dense_1_bias = {&dense_1_bias_array[0], 1, 3, {3, 1, 1, 1, 1}};
	float dense_1_fwork[24] = {0};

	float dense_2_kernel_array[3] = {
		+2.42817879e+00f,
		+2.08857155e+00f,
		+2.20083642e+00f,
	};
	k2c_tensor dense_2_kernel = {&dense_2_kernel_array[0], 2, 3, {3, 1, 1, 1, 1}};
	float dense_2_bias_array[1] = {
		+1.62773621e+00f,
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

void HeartRateService::OnNewHeartRateValue(uint8_t heartRateValue) {
  if (!heartRateMeasurementNotificationEnable)
    return;

  uint8_t med_idx = buffer_1_index >> 1;
  uint8_t qr_idx = buffer_1_index >> 2;
  float median, iqr, max;
  median = buffer_1[med_idx];
  iqr = buffer_1[qr_idx*3] - buffer_1[qr_idx];
  max = buffer_1[buffer_1_index-1];
  float hr_mean = 80.9075072584f;
  float hr_std_inv = 0.0367167158f;
  float median_mean = 139.950683391;
  float median_std_inv = 0.0906305811;
  float iqr_mean = 31.9442277336;
  float iqr_std_inv = 0.0421347519;
  float max_mean = 190.8756786542;
  float max_std_inv = 0.0272782309;

  float normal_hr, normal_median, normal_iqr, normal_max;
  normal_hr = normalize(heartRateValue, hr_mean, hr_std_inv);
  normal_median = normalize(median, median_mean, median_std_inv);
  normal_iqr = normalize(iqr, iqr_mean, iqr_std_inv);
  normal_max = normalize(max, max_mean, max_std_inv);

  heartRateMeasurementBuffer[hr_buffer_index] = uint8_t(normal_hr);
  hr_buffer_index = (hr_buffer_index + 1) % 7;
  acc_data_t window = {normal_median, normal_max, normal_iqr};
  acc_data_buffer[acc_buffer_index] = window;
  acc_buffer_index = (acc_buffer_index + 1) % 3;

  // effectively clears the buffer
  buffer_1_index = 0;

  float input_buffer[16];

  for(int i=0; i<7; i++){
    input_buffer[i] = heartRateMeasurementBuffer[(hr_buffer_index - 1 + i) % 7];
  }

  for(int i=0; i<3; i++){
    input_buffer[7 + i] = acc_data_buffer[(acc_buffer_index-1 + i) % 3].median;
    input_buffer[10 + i] = acc_data_buffer[(acc_buffer_index-1 + i) % 3].max;
    input_buffer[13 + i] = acc_data_buffer[(acc_buffer_index-1 + i) % 3].iqr;
  }

  k2c_tensor input_1_input = {&input_buffer[0], 1, 16, {16, 1, 1, 1, 1}};
  float dense_output_array[1] = {0};
  k2c_tensor dense_output = {&dense_output_array[0], 1, 1, {1, 1, 1, 1, 1}};

  filter_hr(&input_1_input, &dense_output);

  int output = (int)dense_output.array[0];
  uint8_t inference;
  if(output < 0){
  inference = 0;
  }else if(output > 255){
  inference = 255;
  }else{
  inference = (uint8_t)output;
  }
  uint8_t buffer[3] = {0, heartRateValue, inference}; // [0] = flags, [1] = hr value
  auto* om = ble_hs_mbuf_from_flat(buffer, 3);

  uint16_t connectionHandle = nimble.connHandle();

  if (connectionHandle == 0 || connectionHandle == BLE_HS_CONN_HANDLE_NONE) {
    return;
  }

  ble_gattc_notify_custom(connectionHandle, heartRateMeasurementHandle, om);
}

void HeartRateService::OnNewMotionValues(int16_t x, int16_t y, int16_t z) {
  if (!heartRateMeasurementNotificationEnable)
    return;

  // float alpha = 0.29870618761437979596f;
  // float beta = 0.59741237522875959192f;
  // float gamma = 0.93980863517232523127f;
  // float min, med, max, t;

  // int16_t tmp = x >> 15;
  // x = (x ^ tmp) - tmp;
  // tmp = y >> 15;
  // y = (y ^ tmp) - tmp;
  // tmp = z >> 15;
  // z = (z ^ tmp) - tmp;
  float fx = float(x);// /2048.0f;
  float fy = float(y);// /2048.0f;
  float fz = float(z);// /2048.0f;
  float L = sqrt(fx*fx + fy*fy + fz*fz);
  insert_into_buffer_1(L);
}

void HeartRateService::insert_into_buffer_1(float value){
  if(buffer_1_index >= BUFFER_1_LENGTH)
    buffer_1_index = 0;
  buffer_1[buffer_1_index] = value;
  buffer_1_index++;


  // sort buffer_1 with insertion sort from 0 to buffer_1_index
  for(int i = 1; i < buffer_1_index; i++){
    float key = buffer_1[i];
    int j = i - 1;
    while(j >= 0 && buffer_1[j] > key){
      buffer_1[j+1] = buffer_1[j];
      j--;
    }
    buffer_1[j+1] = key;
  }
}


void HeartRateService::SubscribeNotification(uint16_t attributeHandle) {
  if (attributeHandle == heartRateMeasurementHandle)
    heartRateMeasurementNotificationEnable = true;
}

void HeartRateService::UnsubscribeNotification(uint16_t attributeHandle) {
  if (attributeHandle == heartRateMeasurementHandle)
    heartRateMeasurementNotificationEnable = false;
}
