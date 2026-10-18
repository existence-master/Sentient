#include "camera.h"

#include "app_config.h"
#include "driver/gpio.h"
#include "esp_camera.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "camera";
static bool s_present;     // the sensor answered at boot (or camera-off mode, where it is never powered to check)
static bool s_off_mode;    // camera-off mode: set at boot on the glasses, never changed by the engine
static bool s_powered;     // the camera's supply (and with it the privacy LED) is on
static camera_fb_t *s_fb;  // only one capture is in flight at a time

static const camera_config_t CONFIG = {
    .pin_pwdn = CAM_PIN_PWDN,
    .pin_reset = CAM_PIN_RESET,
    .pin_xclk = CAM_PIN_XCLK,
    .pin_sccb_sda = CAM_PIN_SIOD,
    .pin_sccb_scl = CAM_PIN_SIOC,
    .pin_d7 = CAM_PIN_D7,
    .pin_d6 = CAM_PIN_D6,
    .pin_d5 = CAM_PIN_D5,
    .pin_d4 = CAM_PIN_D4,
    .pin_d3 = CAM_PIN_D3,
    .pin_d2 = CAM_PIN_D2,
    .pin_d1 = CAM_PIN_D1,
    .pin_d0 = CAM_PIN_D0,
    .pin_vsync = CAM_PIN_VSYNC,
    .pin_href = CAM_PIN_HREF,
    .pin_pclk = CAM_PIN_PCLK,
    .xclk_freq_hz = CAM_XCLK_HZ,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    .pixel_format = PIXFORMAT_JPEG,   // the engine wants a JPEG; the sensor encodes it for us
    .frame_size = FRAMESIZE_VGA,      // 640x480: 30-60 KB per photo at quality 12
    .jpeg_quality = CAM_JPEG_QUALITY, // lower number = better quality = bigger file
    .fb_count = CAM_FB_COUNT,
    .fb_location = CAMERA_FB_IN_PSRAM, // 8 MB PSRAM: never put JPEG buffers in internal RAM
    .grab_mode = CAMERA_GRAB_LATEST,   // always the newest frame, not a queued stale one
};

// Switches the camera's supply. The privacy LED sits on the same switched rail (README, "Camera privacy"), so it
// is lit exactly while this is on: there is no way to run the camera with the LED dark.
static void power_set(bool on) {
#if CAM_POWER_GPIO >= 0
    gpio_set_level(CAM_POWER_GPIO, on ? CAM_POWER_ON_LEVEL : !CAM_POWER_ON_LEVEL);
#endif
#if CAM_PIN_PWDN >= 0
    if (!on) {
        // OV3660 PWDN is active high: hold the sensor in power-down while it is not in use
        gpio_set_direction(CAM_PIN_PWDN, GPIO_MODE_OUTPUT);
        gpio_set_level(CAM_PIN_PWDN, 1);
    }
#endif
    s_powered = on;
}

static void power_pins_init(void) {
#if CAM_POWER_GPIO >= 0
    gpio_config_t cfg = {
        .pin_bit_mask = 1ULL << CAM_POWER_GPIO,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&cfg);
#else
    ESP_LOGW(TAG, "CAM_POWER_GPIO is not wired: the privacy LED cannot follow the camera's power (README)");
#endif
    power_set(false); // off from the first moment, whatever the pin did during reset
}

// Powers the camera and starts the sensor. On failure everything is powered off again.
static bool power_up(void) {
    if (s_off_mode || CAMERA_HARD_OFF) return false; // never power the camera in camera-off mode
    power_set(true);
    vTaskDelay(pdMS_TO_TICKS(CAM_POWER_SETTLE_MS));
    esp_err_t err = esp_camera_init(&CONFIG);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "camera init failed (0x%x): check the pins in app_config.h and the FPC seating", err);
        power_set(false);
        return false;
    }
    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor != NULL) {
        // OV3660 specifics: the module is usually mounted upside down, and it oversaturates.
        sensor->set_vflip(sensor, 1);
        sensor->set_brightness(sensor, 1);
        sensor->set_saturation(sensor, -2);
    }
    return true;
}

static void power_down(void) {
    if (!s_powered) return;
    esp_camera_deinit(); // stops XCLK and frees the PSRAM frame buffers
    power_set(false);
}

void camera_setup(bool off_mode) {
    s_off_mode = off_mode || CAMERA_HARD_OFF;
    power_pins_init();
    if (CAMERA_HARD_OFF) {
        s_present = false; // built without a camera: camera.photo is not offered
        ESP_LOGI(TAG, "built with CAMERA_HARD_OFF: the camera is never powered");
        return;
    }
    if (s_off_mode) {
        s_present = true; // not powered to check: the board has a camera, it is just switched off
        ESP_LOGW(TAG, "camera-off mode is on: the camera stays unpowered and photos are refused");
        return;
    }
    s_present = power_up(); // one probe at boot: the LED blinks once when the camera works
    power_down();
    if (s_present) ESP_LOGI(TAG, "camera ready (OV3660, VGA, quality %d), powered only for photos", CAM_JPEG_QUALITY);
}

bool camera_available(void) {
    return s_present;
}

bool camera_off_mode(void) {
    return s_off_mode;
}

bool camera_capture(const uint8_t **data, size_t *len, int *width, int *height) {
    if (s_off_mode) {
        ESP_LOGW(TAG, "photo refused: camera-off mode is on");
        return false;
    }
    if (!s_present) return false;
    camera_release(); // make sure the previous frame went back and the camera is off
    if (!power_up()) return false;
    for (int i = 0; i < CAM_WARMUP_FRAMES; i++) { // the first frames after power-up are dark while exposure settles
        camera_fb_t *warm = esp_camera_fb_get();
        if (warm != NULL) esp_camera_fb_return(warm);
    }
    s_fb = esp_camera_fb_get();
    if (s_fb == NULL) {
        ESP_LOGE(TAG, "capture failed");
        power_down();
        return false;
    }
    *data = s_fb->buf;
    *len = s_fb->len;
    *width = (int)s_fb->width;
    *height = (int)s_fb->height;
    ESP_LOGI(TAG, "photo %dx%d, %u bytes", *width, *height, (unsigned)*len);
    return true;
}

void camera_release(void) {
    if (s_fb != NULL) {
        esp_camera_fb_return(s_fb);
        s_fb = NULL;
    }
    power_down(); // the camera (and its LED) is off again as soon as the photo is handed over
}
