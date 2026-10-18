// OV3660 capture for `camera.photo` (docs/NODES.md sections 11.3 and 11.7).
//
// The camera is powered only while a photo is taken: power on, start the sensor, grab one JPEG, stop the sensor,
// power off. Wire the privacy LED on the camera's switched supply (CAM_POWER_GPIO in app_config.h), so it is lit
// exactly while the camera has power. In camera-off mode the camera is never powered at all.
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Call once at boot, before anything else touches the camera. `off_mode`: camera-off mode, set on the glasses
// themselves (main.c) and never by the engine. Outside camera-off mode the sensor is probed once (the LED blinks).
void camera_setup(bool off_mode);

bool camera_available(void); // this board has a camera (advertise camera.photo), even while camera-off mode is on
bool camera_off_mode(void);  // camera-off mode is on: every photo is refused and the camera stays unpowered

// Powers the camera and captures one JPEG. False in camera-off mode (without powering anything) or on failure.
// The buffer belongs to the driver: send it, then call camera_release(), which also powers the camera off.
// It lives in PSRAM, so it can be handed to the websocket client without copying.
bool camera_capture(const uint8_t **data, size_t *len, int *width, int *height);
void camera_release(void);
