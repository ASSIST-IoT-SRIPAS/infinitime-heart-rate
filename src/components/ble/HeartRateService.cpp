#include "components/ble/HeartRateService.h"
#include "components/heartrate/HeartRateController.h"
#include "components/motion/MotionController.h"
#include "components/ble/NimbleController.h"
#include <nrf_log.h>
#include <hal/nrf_rtc.h>
#include <cmath>
#include <time.h>

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
  motionController.setHeartRateService(this);
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

  heartRateMeasurementBuffer[hr_buffer_index] = normal_hr;
  hr_buffer_index = (hr_buffer_index + 1) % 5;
  acc_data_t window = {normal_median, normal_max, normal_iqr};
  acc_data_buffer[acc_buffer_index] = window;
  acc_buffer_index = (acc_buffer_index + 1) % 3;

  float input_buffer[14];

  for(int i=0; i<5; i++){
    input_buffer[i] = heartRateMeasurementBuffer[(hr_buffer_index + i) % 5];
  }

  for(int i=0; i<3; i++){
    input_buffer[5 + i] = acc_data_buffer[(acc_buffer_index + i) % 3].median;
    input_buffer[8 + i] = acc_data_buffer[(acc_buffer_index + i) % 3].max;
    input_buffer[11 + i] = acc_data_buffer[(acc_buffer_index + i) % 3].iqr;
  }

  k2c_tensor input_1_input = {&input_buffer[0], 1, 14, {14, 1, 1, 1, 1}};
  float dense_output_array[1] = {0};
  k2c_tensor dense_output = {&dense_output_array[0], 1, 1, {1, 1, 1, 1, 1}};
  filter_hr(&input_1_input, &dense_output);

  float output = dense_output.array[0];
  uint8_t inference;
  if(output < 0){
  inference = 0;
  }else if(output > 255){
  inference = 255;
  }else{
  inference = (uint8_t)output;
  }
  uint8_t buffer[2] = {0, inference}; // [0] = flags, [1] = hr value 
  auto* om = ble_hs_mbuf_from_flat(buffer, 2);

  uint16_t connectionHandle = nimble.connHandle();

  if (connectionHandle == 0 || connectionHandle == BLE_HS_CONN_HANDLE_NONE) {
    return;
  }

  ble_gattc_notify_custom(connectionHandle, heartRateMeasurementHandle, om);
   
  // effectively clears the buffer
  buffer_1_index = 0;
}

void HeartRateService::OnNewMotionValues(int16_t x, int16_t y, int16_t z) {
  if (!heartRateMeasurementNotificationEnable)
    return;

  float fx = float(x);
  float fy = float(y);
  float fz = float(z);
  float L = sqrt(fx*fx + fy*fy + fz*fz);
  insert_into_buffer_1(L);
}

void HeartRateService::insert_into_buffer_1(float value){
  if(buffer_1_index >= BUFFER_1_LENGTH)
    // this never happens in practice
    // , as the buffer has a lot of extra length
    buffer_1_index = 0;

  int i = buffer_1_index - 1;
  while(i >= 0 && buffer_1[i] > value){
      buffer_1[i+1] = buffer_1[i];
      i--;
  }
  buffer_1[i+1] = value;
  buffer_1_index++;

  // todo change to inserting into a sorted array
  // for(int i = 1; i < buffer_1_index; i++){
  //   float key = buffer_1[i];
  //   int j = i - 1;
  //   while(j >= 0 && buffer_1[j] > key){
  //     buffer_1[j+1] = buffer_1[j];
  //     j--;
  //   }
  //   buffer_1[j+1] = key;
  // }
  
}


void HeartRateService::SubscribeNotification(uint16_t attributeHandle) {
  if (attributeHandle == heartRateMeasurementHandle)
    heartRateMeasurementNotificationEnable = true;
}

void HeartRateService::UnsubscribeNotification(uint16_t attributeHandle) {
  if (attributeHandle == heartRateMeasurementHandle)
    heartRateMeasurementNotificationEnable = false;
}
