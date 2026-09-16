// Every pin, buffer size and tunable for the Sentient glasses board in one place.
// Protocol details: docs/NODES.md (section 11 describes this exact hardware).
#pragma once

// ---------------------------------------------------------------- identity
#define NODE_NAME "Sentient Glasses" // shown in the Sentient Devices list; the user can rename it there
#define NODE_KIND "glasses"          // phone | glasses | watch | custom
#define NODE_APP_VERSION "0.1.0"

// ---------------------------------------------------------------- Wi-Fi
#define WIFI_SSID "your-network"
#define WIFI_PASSWORD "your-password"
#define WIFI_BACKOFF_MIN_MS 1000  // first retry after a drop
#define WIFI_BACKOFF_MAX_MS 30000 // cap, same curve as docs/NODES.md section 8

// ---------------------------------------------------------------- engine
// Leave ENGINE_HOST empty to discover the engine over mDNS (_sentient._tcp.local.).
// Fill both in to skip discovery (useful on a network that blocks multicast).
#define ENGINE_HOST ""
#define ENGINE_PORT 7778

// SHA-256 of the engine's TLS certificate: Sentient shows it under Devices
// (GET /api/nodes/lan -> "fingerprint"), 64 lowercase hex characters.
// Empty = trust the fingerprint from the mDNS TXT record on first pairing (weaker, see docs/NODES.md 2).
#define ENGINE_FINGERPRINT ""

// Pairing code from Sentient (Devices > Pair a device). Only needed once: after a successful
// pairing the token lives in NVS and this is ignored. Leave empty once paired.
#define PAIRING_CODE ""

// ---------------------------------------------------------------- microphone: INMP441 on I2S0
#define MIC_I2S_PORT I2S_NUM_0
#define MIC_PIN_SCK 4 // INMP441 SCK  (bit clock)
#define MIC_PIN_WS 5  // INMP441 WS   (word select / LR clock)
#define MIC_PIN_SD 6  // INMP441 SD   (data out of the mic, into the ESP32)
// INMP441 L/R pin tied to GND -> it drives the LEFT slot.
#define MIC_SAMPLE_RATE 16000 // what /ws/voice expects; do not change without resampling
#define MIC_GAIN_SHIFT 14     // raw 32-bit slot -> int16: >>16 is "correct" but quiet; 14 adds ~12 dB
#define MIC_DMA_FRAMES 240    // frames per DMA buffer (~15 ms at 16 kHz)
#define MIC_DMA_BUFFERS 4     // ~60 ms of slack before samples are dropped
#define MIC_CHUNK_SAMPLES 640 // 40 ms per websocket frame = 1280 bytes of PCM16

// ---------------------------------------------------------------- speaker: MAX98357A on I2S1
#define AMP_I2S_PORT I2S_NUM_1
#define AMP_PIN_BCLK 15 // MAX98357A BCLK
#define AMP_PIN_LRC 16  // MAX98357A LRC (word select)
#define AMP_PIN_DIN 7   // MAX98357A DIN (data from the ESP32)
// MAX98357A SD pin: tie high for "always on" (or wire to a GPIO and set AMP_PIN_SHUTDOWN).
#define AMP_PIN_SHUTDOWN -1        // -1 = not wired
#define AMP_DEFAULT_SAMPLE_RATE 16000 // reconfigured per reply from the engine's sample_rate
#define AMP_DMA_FRAMES 240
#define AMP_DMA_BUFFERS 6            // playback tolerates more latency than capture
#define AUDIO_RING_BYTES (32 * 1024) // ~1 s at 16 kHz PCM16: enough to cover Wi-Fi jitter

// ---------------------------------------------------------------- camera: OV3660
// These are the pins of the common ESP32-S3 camera carrier boards (Freenove / ESP32-S3-EYE style).
// VERIFY THEM against the glasses schematic before the first flash: wrong pins usually show up as
// "Camera probe failed with error 0x105".
#define CAM_PIN_PWDN -1
#define CAM_PIN_RESET -1
#define CAM_PIN_XCLK 15
#define CAM_PIN_SIOD 4 // SCCB data
#define CAM_PIN_SIOC 5 // SCCB clock
#define CAM_PIN_D7 16
#define CAM_PIN_D6 17
#define CAM_PIN_D5 18
#define CAM_PIN_D4 12
#define CAM_PIN_D3 10
#define CAM_PIN_D2 8
#define CAM_PIN_D1 9
#define CAM_PIN_D0 11
#define CAM_PIN_VSYNC 6
#define CAM_PIN_HREF 7
#define CAM_PIN_PCLK 13
#define CAM_XCLK_HZ 20000000
#define CAM_JPEG_QUALITY 12 // 0 best .. 63 worst; 12 at VGA gives a 30-60 KB photo
#define CAM_FB_COUNT 2

// ---------------------------------------------------------------- button
#define BUTTON_GPIO 0            // internal pull-up, button shorts to GND
#define BUTTON_DEBOUNCE_MS 30
#define BUTTON_LONG_PRESS_MS 2000

// ---------------------------------------------------------------- battery
#define BATTERY_ADC_CHANNEL ADC_CHANNEL_3 // ADC1 channel (GPIO4 on the S3 is ADC1_CH3)
#define BATTERY_DIVIDER 2.0f              // two equal resistors halve the cell voltage
#define BATTERY_FULL_MV 4200
#define BATTERY_EMPTY_MV 3300
#define BATTERY_REPORT_INTERVAL_MS (5 * 60 * 1000)
#define BATTERY_REPORT_DELTA 2 // also report when the percentage moves this much

// ---------------------------------------------------------------- sockets
#define WS_TEXT_MAX 4096        // biggest JSON we accept; engine messages to devices are far smaller
#define WS_BUFFER_SIZE 2048     // websocket client RX/TX buffer (keeps TLS records small)
#define WS_SEND_TIMEOUT_MS 5000
#define VOICE_IDLE_CLOSE_MS 20000 // close /ws/voice after this much silence to save power
#define NODE_TASK_STACK 6144
