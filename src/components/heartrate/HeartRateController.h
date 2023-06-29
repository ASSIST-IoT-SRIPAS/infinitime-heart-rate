#pragma once

#include <cstdint>
#include <components/ble/HeartRateService.h>

namespace Pinetime {
  namespace Applications {
    class HeartRateTask;
  }
  namespace System {
    class SystemTask;
  }
  namespace Controllers {
    class HeartRateController {
    public:
      enum class States { Stopped, NotEnoughData, NoTouch, Running };

      HeartRateController() = default;
      void Start();
      void Stop();
      void Update(States newState, uint8_t heartRate);
      void UpdateML(uint8_t mlheartRate);


      void SetHeartRateTask(Applications::HeartRateTask* task);
      States State() const {
        return state;
      }
      uint8_t HeartRate() const {
        return heartRate;
      }

      uint8_t MLHeartRate() const {
        return mlheartRate;
      }

      void SetService(Pinetime::Controllers::HeartRateService* service);

    private:
      Applications::HeartRateTask* task = nullptr;
      States state = States::Stopped;
      uint8_t heartRate = 0;
      uint8_t mlheartRate = 0;
      Pinetime::Controllers::HeartRateService* service = nullptr;
    };
  }
}
