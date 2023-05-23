# About

This repository is a fork of the [InfiniTime](https://github.com/InfiniTimeOrg/InfiniTime) project, developed for the purposes of Pilot 2: Smart Safety of workers of the [ASSIST-IoT](https://assist-iot.eu) Horizon 2020 project.
It deploys a neural network based model to improve the quality of readings from the onboard heart rate sensor.
The model has been developed keeping in mind the principles of frugality, which ensures that all existing InfiniTime functionalities operate smoothly.

## Contents

- The main implementation of the model is in the [`HeartRateService.cpp` file](https://github.com/ASSIST-IoT-SRIPAS/infinitime-heart-rate/blob/main/src/components/ble/HeartRateService.cpp).
- The model itself is in the [`model` directory](https://github.com/ASSIST-IoT-SRIPAS/infinitime-heart-rate/tree/main/model).
    - `model.h5` is the model in the Keras H5 format.
    - `filter_hr.c` and `filter_hr.h` is the model converted to C by the keras2c library.

## Authors

- [Kajetan Rachwał](https://orcid.org/0000-0003-1524-7877) – implementation of the model and the preprocessing stages in InfiniTime, integration
- [Piotr Sowiński](https://orcid.org/0000-0002-2543-9461) – training the model and designing the preprocessing stages

## Used libraries

The implementation of the model uses the [keras2c library](https://f0uriest.github.io/keras2c/index.html) by Rory Conlin. keras2c is licensed under the MIT license.

## Acknowledgements

This work is part of the [ASSIST-IoT project](https://assist-iot.eu/) that has received funding from the EU’s Horizon 2020 research and innovation programme under grant agreement No 957258.

----

What follows is the description of the InfiniTime project, which includes instructions for building and deployment of the solution on the PineTime smartwatch.


# [InfiniTime](https://github.com/InfiniTimeOrg/InfiniTime)

![InfiniTime logo](doc/logo/infinitime-logo-small.jpg "InfiniTime Logo")

Fast open-source firmware for the [PineTime smartwatch](https://www.pine64.org/pinetime/) with many features, written in modern C++.

## New to InfiniTime?

- [Getting started with InfiniTime](doc/gettingStarted/gettingStarted-1.0.md)
- [Updating the software](doc/gettingStarted/updating-software.md)
- [About the firmware and bootloader](doc/gettingStarted/about-software.md)

### Companion apps

- [Gadgetbridge](https://gadgetbridge.org/) (Android)
- [AmazFish](https://openrepos.net/content/piggz/amazfish/) (SailfishOS)
- [Siglo](https://github.com/alexr4535/siglo) (Linux)
- [InfiniLink](https://github.com/InfiniTimeOrg/InfiniLink) (iOS) **[Looking for a new maintainer]**
- [ITD](https://gitea.arsenm.dev/Arsen6331/itd) (Linux)
- [WatchMate](https://github.com/azymohliad/watchmate) (Linux)

***Note** : We removed mentions to NRFConnect as this app is closed source and recent versions do not work anymore with InfiniTime (the last version known to work is 4.24.3). If you used NRFConnect in the past, we recommend you switch to [Gadgetbridge](https://gadgetbridge.org/).*

## Development

- [InfiniTime Vision](doc/InfiniTimeVision.md)
- [Rough structure of the code](doc/code/Intro.md)
- [How to implement an application](doc/code/Apps.md)
- [Generate the fonts and symbols](src/displayapp/fonts/README.md)
- [Tips on designing an app UI](doc/ui_guidelines.md)
- [Bootloader, OTA and DFU](bootloader/README.md)
- [External resources](doc/ExternalResources.md)

### Contributing

- [How to contribute?](CONTRIBUTING.md)
- [Coding conventions](doc/coding-convention.md)

### Build, flash and debug

- [InfiniTime simulator](https://github.com/InfiniTimeOrg/InfiniSim)
- [Build the project](doc/buildAndProgram.md)
- [Build the project with Docker](doc/buildWithDocker.md)
- [Build the project with VSCode](doc/buildWithVScode.md)
- [Flash the firmware using OpenOCD and STLinkV2](doc/openOCD.md)
- [Flash the firmware using SWD interface](doc/SWD.md)
- [Flash the firmware using JLink](doc/jlink.md)
- [Flash the firmware using GDB](doc/gdb.md)
- [Stub using NRF52-DK](doc/PinetimeStubWithNrf52DK.md)

### API

- [BLE implementation and API](doc/ble.md)

### Architecture and technical topics

- [Memory analysis](doc/MemoryAnalysis.md)

### Project management

- [Maintainer's guide](doc/maintainer-guide.md)
- [Versioning](doc/versioning.md)
- [Project branches](doc/branches.md)
- [Files included in the release notes](doc/filesInReleaseNotes.md)
- [Files needed by the factory](doc/files-needed-by-factory.md)

## Licenses

This project is released under the GNU General Public License version 3 or, at your option, any later version.

It integrates the following projects:

- RTOS : **[FreeRTOS](https://freertos.org)** under the MIT license
- UI : **[LittleVGL/LVGL](https://lvgl.io/)** under the MIT license
- BLE stack : **[NimBLE](https://github.com/apache/mynewt-nimble)** under the Apache 2.0 license
- Font : **[Jetbrains Mono](https://www.jetbrains.com/fr-fr/lp/mono/)** under the Apache 2.0 license

## Credits

I’m not working alone on this project. First, many people create PR for this project. Then, there is the whole #pinetime community : a lot of people all around the world who are hacking, searching, experimenting and programming the Pinetime. We exchange our ideas, experiments and code in the chat rooms and forums.

Here are some people I would like to highlight:

- [Atc1441](https://github.com/atc1441/) : He works on an Arduino based firmware for the Pinetime and many other smartwatches based on similar hardware. He was of great help when I was implementing support for the BMA421 motion sensor and I²C driver.
- [Koen](https://github.com/bosmoment) : He’s working on a firmware based on RiotOS. He integrated similar libs as me : NimBLE, LittleVGL,… His help was invaluable too!
- [Lup Yuen Lee](https://github.com/lupyuen) : He is everywhere: he works on a Rust firmware, builds a MCUBoot based bootloader for the Pinetime, designs a Flutter based companion app for smartphones and writes a lot of articles about the Pinetime!
