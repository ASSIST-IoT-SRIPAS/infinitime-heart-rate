#pragma once
#define min // workaround: nimble's min/max macros conflict with libstdc++
#define max
#include <host/ble_gap.h>
#include <atomic>
#undef max
#undef min

#define BUFFER_1_LENGTH 200
#define K2C_MAX_NDIM 5

namespace Pinetime {
  namespace System {
    class SystemTask;
  }
  namespace Controllers {
    class HeartRateController;
    class MotionController;
    class HeartRateService {

    typedef struct acc_data {
      float median = {0};
      float max = {0};
      float iqr = {0};
    } acc_data_t;


    public:
      HeartRateService(Pinetime::System::SystemTask& system, Controllers::HeartRateController& heartRateController, Controllers::MotionController& motionController);
      void Init();
      int OnHeartRateRequested(uint16_t connectionHandle, uint16_t attributeHandle, ble_gatt_access_ctxt* context);
      void OnNewHeartRateValue(uint8_t hearRateValue);
      void OnNewMotionValues(int16_t x, int16_t y, int16_t z);

      void SubscribeNotification(uint16_t connectionHandle, uint16_t attributeHandle);
      void UnsubscribeNotification(uint16_t connectionHandle, uint16_t attributeHandle);

    private:
      Pinetime::System::SystemTask& system;
      Controllers::HeartRateController& heartRateController;
      Controllers::MotionController& motionController;
      static constexpr uint16_t heartRateServiceId {0x180D};
      static constexpr uint16_t heartRateMeasurementId {0x2A37};

      static constexpr ble_uuid16_t heartRateServiceUuid {.u {.type = BLE_UUID_TYPE_16}, .value = heartRateServiceId};

      static constexpr ble_uuid16_t heartRateMeasurementUuid {.u {.type = BLE_UUID_TYPE_16}, .value = heartRateMeasurementId};

      struct ble_gatt_chr_def characteristicDefinition[2];
      struct ble_gatt_svc_def serviceDefinition[2];

      uint16_t heartRateMeasurementHandle;
      uint16_t motionValuesHandle;

      float heartRateMeasurementBuffer[5] = {0x00, 0x00, 0x00, 0x00, 0x00};
      uint8_t hr_buffer_index = 0;

      acc_data_t acc_data_buffer[3];
      uint8_t acc_buffer_index = 0;


      float buffer_1[BUFFER_1_LENGTH];
      uint8_t buffer_1_index = 0;


      void insert_into_buffer_1(float value);
      float normalize(float value, float mean, float std_inv);

      std::atomic_bool heartRateMeasurementNotificationEnable {false};
    };
  }
}
