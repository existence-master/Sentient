// Capability handlers: what this board actually does when the engine asks (docs/NODES.md section 5).
#pragma once
#include <stdint.h>

// Runs one invoke and sends its result. `params_json` is the raw params object. `stops_when_queued` is
// caps_photo_stops() when the invoke arrived: a camera.photo is dropped if Sentient turned the camera off since.
void caps_handle_invoke(const char *id, const char *capability, const char *params_json, uint32_t stops_when_queued);

// Where display.text / display.card / notify.show end up. The default prints to the serial
// console; wire an OLED here (see the README) if the glasses have one.
void caps_show(const char *title, const char *text);

// Sentient turned the camera off (stop_sensor): every photo asked for before now, queued or being taken, is dropped
// and never sent. caps_photo_stops() counts these, so a queued invoke can tell whether one came after it.
void caps_cancel_photo(void);
uint32_t caps_photo_stops(void);
