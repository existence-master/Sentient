#include "battery.h"

#include "app_config.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_log.h"

static const char *TAG = "battery";
static adc_oneshot_unit_handle_t s_adc;
static adc_cali_handle_t s_cali;
static bool s_ready;

void battery_init(void) {
    adc_oneshot_unit_init_cfg_t unit = {.unit_id = ADC_UNIT_1};
    if (adc_oneshot_new_unit(&unit, &s_adc) != ESP_OK) {
        ESP_LOGW(TAG, "no ADC unit; battery reporting disabled");
        return;
    }
    adc_oneshot_chan_cfg_t chan = {.atten = ADC_ATTEN_DB_12, .bitwidth = ADC_BITWIDTH_DEFAULT};
    ESP_ERROR_CHECK(adc_oneshot_config_channel(s_adc, BATTERY_ADC_CHANNEL, &chan));

    // Curve fitting is the calibration scheme available on the ESP32-S3.
    adc_cali_curve_fitting_config_t cali = {
        .unit_id = ADC_UNIT_1,
        .atten = ADC_ATTEN_DB_12,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };
    if (adc_cali_create_scheme_curve_fitting(&cali, &s_cali) != ESP_OK) {
        ESP_LOGW(TAG, "ADC not calibrated; readings are approximate");
        s_cali = NULL;
    }
    s_ready = true;
}

int battery_millivolts(void) {
    if (!s_ready) return -1;
    int raw = 0, mv = 0;
    if (adc_oneshot_read(s_adc, BATTERY_ADC_CHANNEL, &raw) != ESP_OK) return -1;
    if (s_cali != NULL) {
        if (adc_cali_raw_to_voltage(s_cali, raw, &mv) != ESP_OK) return -1;
    } else {
        mv = raw * 3100 / 4095; // rough fallback for ADC_ATTEN_DB_12
    }
    return (int)(mv * BATTERY_DIVIDER);
}

int battery_percent(void) {
    int mv = battery_millivolts();
    if (mv < 0) return -1;
    if (mv <= BATTERY_EMPTY_MV) return 0;
    if (mv >= BATTERY_FULL_MV) return 100;
    // Linear between empty and full. Good enough for a status pill; a real gauge would use a
    // discharge curve for the specific cell.
    return (mv - BATTERY_EMPTY_MV) * 100 / (BATTERY_FULL_MV - BATTERY_EMPTY_MV);
}
