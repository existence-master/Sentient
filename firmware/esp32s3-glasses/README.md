# Sentient glasses firmware (ESP32-S3)

Reference firmware for the Sentient smart glasses: it pairs with the Sentient engine on your
network, answers `invoke` requests (photo, text, notification, speech, battery) and streams
hold-to-talk audio to the assistant.

The protocol is in `docs/NODES.md` (section 11 describes this exact board). The engine side is
`sentient/nodes/`, and `sentient node` on a PC is a second reference implementation you can compare
against when something here misbehaves.

> **Nothing in this folder has been compiled or flashed yet.** It was written against the ESP-IDF
> 5.3 APIs and the protocol the engine's tests cover, but no ESP32-S3 has run it. Treat the first
> flash as bring-up: expect to fix pin numbers and check every "untested" note below.

## Hardware

| Part | Role | Notes |
|---|---|---|
| ESP32-S3 (8 MB PSRAM) | everything | PSRAM is required: JPEG frame buffers and Wi-Fi buffers live there |
| INMP441 | I2S microphone | `L/R` to GND, so it drives the left slot |
| MAX98357A | I2S amplifier + speaker | `SD` high for always-on, `GAIN` floating = 9 dB |
| OV3660 | camera | JPEG straight from the sensor, VGA |
| push button | talk button | to GND, the internal pull-up is enabled in firmware |
| battery divider | battery percentage | two equal resistors into an ADC1 pin |

Default wiring (all of it is `#define`d in `main/app_config.h`, change it to match the board):

```
INMP441      SCK -> GPIO4      WS  -> GPIO5      SD  -> GPIO6      L/R -> GND
MAX98357A    BCLK-> GPIO15     LRC -> GPIO16     DIN -> GPIO7      SD  -> 3V3
Button       GPIO0 -> button -> GND
Battery      divider -> GPIO4 (ADC1_CH3)        <-- clashes with the mic SCK default, move one of them
OV3660       XCLK/SIOD/SIOC/D0-D7/VSYNC/HREF/PCLK: see CAM_PIN_* in app_config.h
```

**Untested / must check:** the camera pins are the ones used by the common ESP32-S3 camera carrier
boards, not by your glasses PCB, and the battery ADC default collides with the microphone clock.
Take both from the schematic before the first build.

## Why ESP-IDF and not Arduino

The board needs three things that decide it: a websocket client that can send **binary frames**
(photos and microphone audio), **certificate pinning** against a self-signed certificate, and control
over where large buffers live (PSRAM). On Arduino you would still end up pulling in
`esp_websocket_client` and `esp32-camera`, which are ESP-IDF components, and no Arduino websocket
library exposes the peer certificate needed for pinning. ESP-IDF gives all three directly, and the
`i2s_std` driver handles two I2S peripherals at once (capture and playback) without extra work.
If you prefer the Arduino IDE, this code can be built as an Arduino-as-an-ESP-IDF-component project
without changes to the files in `main/`.

## Build and flash

Install ESP-IDF **v5.3 or newer** ([setup guide](https://docs.espressif.com/projects/esp-idf/en/v5.3/esp32s3/get-started/)),
then from this folder:

```bash
# 1. edit main/app_config.h: WIFI_SSID, WIFI_PASSWORD, ENGINE_FINGERPRINT, PAIRING_CODE, pins
idf.py set-target esp32s3
idf.py build                       # first build downloads the managed components below
idf.py -p COM5 flash monitor       # Windows; /dev/ttyUSB0 or /dev/ttyACM0 on Linux, /dev/cu.* on macOS
```

Managed components (pulled automatically, pinned in `main/idf_component.yml`):

| component | version | why |
|---|---|---|
| `espressif/esp_websocket_client` | ^1.2.0 | both sockets, text and binary frames |
| `espressif/esp32-camera` | ^2.0.9 | OV3660 driver, JPEG in PSRAM |
| `espressif/mdns` | ^1.2.0 | finds `_sentient._tcp.local.` |

`cJSON`, `esp-tls`, `mbedtls`, `nvs_flash` and the I2S/ADC drivers ship with ESP-IDF.

## First run (pairing)

1. In Sentient: **Settings › Devices**, turn on *Accept devices on your network*.
2. **Devices › Pair a device** shows a 6-digit code and the certificate fingerprint.
3. Put both into `main/app_config.h`:
   ```c
   #define ENGINE_FINGERPRINT "3f9a…"   // 64 hex characters from the Devices screen
   #define PAIRING_CODE "482913"        // valid for 10 minutes, single use
   ```
4. Flash. The serial log should show `pinned certificate …`, then `paired; token saved to NVS`, then
   `connected to Sentient, keepalive 60s`, and the glasses appear in the Devices list.
5. Clear `PAIRING_CODE` (back to `""`) for later builds: the token in NVS is what is used from now on.
   `idf.py erase-flash` wipes the token and the pinned certificate, so pairing starts over.

What works after pairing: "take a photo with my glasses", "show 'turn left' on my glasses",
notifications, battery in the Devices list, and holding the button to talk to the assistant.

## What each file does

| file | contents |
|---|---|
| `main/app_config.h` | every pin, buffer size and tunable, with comments |
| `main/main.c` | start-up order, engine lookup, certificate pinning, button wiring, battery reports |
| `main/net.c` | Wi-Fi join and the 1→30 s reconnect backoff |
| `main/discovery.c` | mDNS query for `_sentient._tcp.local.`, reads the TXT fingerprint |
| `main/cert_pin.c` | fetches the engine certificate, checks its SHA-256, converts it to PEM |
| `main/storage.c` | NVS: token, pinned certificate, last engine address |
| `main/node_client.c` | `/ws/node`: hello/welcome, invoke queue, results, events, keepalive, reconnects |
| `main/caps.c` | what each capability actually does, including the photo binary frame |
| `main/voice.c` | `/ws/voice?node_token=`: streams the microphone, plays replies |
| `main/audio.c` | I2S capture (INMP441) and playback (MAX98357A) with a ring buffer |
| `main/camera.c` | OV3660 setup and JPEG capture |
| `main/button.c`, `main/battery.c` | debounced button polling, ADC battery percentage |

## Memory notes

Two TLS sessions (node + voice) are the main cost, roughly 35-45 KB of heap each with default
buffers. `sdkconfig.defaults` trims that with `CONFIG_MBEDTLS_DYNAMIC_BUFFER=y` and 4 KB/2 KB TLS
record buffers, and pushes Wi-Fi and LWIP buffers into PSRAM. The voice socket is opened on the first
button press and closed after 20 s of silence (`VOICE_IDLE_CLOSE_MS`), so most of the time only one
session is live. Photos are sent straight from the PSRAM frame buffer, so a 60 KB JPEG costs no extra
internal RAM.

## Adding a small OLED

`caps_show()` in `main/caps.c` is the single place display text arrives. It prints to the serial
console. For an SSD1306, initialise it in `app_main()` and draw inside `caps_show()`. Keep the draw
call short: it runs on the invoke worker task, which the engine is waiting on.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `Camera probe failed with error 0x105` | wrong `CAM_PIN_*` values or a loose FPC. Check the schematic, reseat the connector |
| `no Sentient engine answered on this network` | *Accept devices on your network* is off, the PC firewall blocks port 7778, or the AP blocks multicast. Set `ENGINE_HOST`/`ENGINE_PORT` to skip mDNS |
| `FINGERPRINT MISMATCH` | you are talking to a different machine, or Sentient's certificate was regenerated. Re-read the fingerprint from Devices; if it is genuinely new, `idf.py erase-flash` and pair again |
| `engine error [bad_code]` | the code expired (10 minutes) or was already used. Generate a new one |
| `engine error [bad_token]` | the device was removed in Sentient. Set a new `PAIRING_CODE` and reflash |
| `engine error [rate_limited]` | five wrong codes in a minute. Wait a minute |
| connects, then drops after ~3 minutes | the keepalive ping is not going out. Check the `node_ping` timer and that `welcome` was parsed |
| microphone is silence or only noise | `L/R` not at GND, wrong `MIC_PIN_*`, or `MIC_GAIN_SHIFT` off. Try 12 (louder) or 16 (quieter) |
| speech is distorted or too fast/slow | the reply sample rate was not applied. Check `audio_play_begin()` logs "playback at N Hz" and that N matches the engine |
| speech stutters | Wi-Fi is slow: raise `AUDIO_RING_BYTES`, or move closer to the AP |
| `playback buffer full, dropped N bytes` | the speaker cannot keep up with the network; same fix |
| photo arrives corrupted | another message was sent between the result and its binary frame. All sends go through `node_client_send_result_json()` for exactly this reason |
| heap allocation failures during a call | close the voice socket sooner (`VOICE_IDLE_CLOSE_MS`), or drop the camera to `FRAMESIZE_QVGA` |
| reboots with a stack overflow | raise `NODE_TASK_STACK` or the stack of the task named in the panic |

## Known gaps

- **Untested on hardware** (see the top of this file): pin defaults, the INMP441 gain shift, the ADC
  battery mapping and the OV3660 tuning are all first guesses.
- The engine does not implement `audio_format: "pcm16"` on the voice socket yet, so replies arrive as
  WAV. `voice.c` handles both: it detects the `RIFF` header, reads the sample rate from it and skips
  the 44 bytes. Nothing needs to change here when the engine adds raw PCM.
- No OTA updates: the partition table has a single app slot.
- No deep sleep yet. `docs/NODES.md` section 11.5 describes the intended wake-on-button flow.
- `display.card` images are ignored, and `clipboard.*`, `location.get` and `screen.capture` are not
  advertised: this board has no screen worth drawing photos on, no clipboard and no GPS.
