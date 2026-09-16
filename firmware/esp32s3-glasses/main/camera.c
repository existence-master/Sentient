#include "camera.h"

#include "app_config.h"
#include "esp_camera.h"
#include "esp_log.h"

static const char *TAG = "camera";
static bool s_ready;
static camera_fb_t *s_fb; // only one capture is in flight at a time

bool camera_start(void) {
    camera_config_t config = {
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
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "camera init failed (0x%x): check the pins in app_config.h and the FPC seating", err);
        return false;
    }
    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor != NULL) {
        // OV3660 specifics: the module is usually mounted upside down, and it oversaturates.
        sensor->set_vflip(sensor, 1);
        sensor->set_brightness(sensor, 1);
        sensor->set_saturation(sensor, -2);
    }
    s_ready = true;
    ESP_LOGI(TAG, "camera ready (OV3660, VGA, quality %d)", CAM_JPEG_QUALITY);
    return true;
}

bool camera_available(void) {
    return s_ready;
}

bool camera_capture(const uint8_t **data, size_t *len, int *width, int *height) {
    if (!s_ready) return false;
    camera_release(); // make sure the previous frame went back to the driver
    s_fb = esp_camera_fb_get();
    if (s_fb == NULL) {
        ESP_LOGE(TAG, "capture failed");
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
}
