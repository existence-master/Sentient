// Capability handlers: what this board actually does when the engine asks (docs/NODES.md section 5).
#pragma once

// Runs one invoke and sends its result. `params_json` is the raw params object.
void caps_handle_invoke(const char *id, const char *capability, const char *params_json);

// Where display.text / display.card / notify.show end up. The default prints to the serial
// console; wire an OLED here (see the README) if the glasses have one.
void caps_show(const char *title, const char *text);
