#include "net.h"

#include <string.h>

#include "app_config.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_random.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

static const char *TAG = "net";
static EventGroupHandle_t s_events;
static const int BIT_CONNECTED = BIT0;
static int s_backoff_ms = WIFI_BACKOFF_MIN_MS;

// Retry with 1, 2, 4 ... 30 s plus up to 50% jitter, so a router reboot does not get a
// thundering herd of devices and the radio stays off between attempts.
static void schedule_retry(void) {
    int jitter = (int)(esp_random() % (uint32_t)(s_backoff_ms / 2 + 1));
    int wait_ms = s_backoff_ms + jitter;
    ESP_LOGI(TAG, "Wi-Fi retry in %d ms", wait_ms);
    vTaskDelay(pdMS_TO_TICKS(wait_ms));
    s_backoff_ms = s_backoff_ms * 2 > WIFI_BACKOFF_MAX_MS ? WIFI_BACKOFF_MAX_MS : s_backoff_ms * 2;
    esp_wifi_connect();
}

static void on_event(void *arg, esp_event_base_t base, int32_t id, void *data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(s_events, BIT_CONNECTED);
        ESP_LOGW(TAG, "Wi-Fi disconnected");
        schedule_retry();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *got = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "connected, ip " IPSTR, IP2STR(&got->ip_info.ip));
        s_backoff_ms = WIFI_BACKOFF_MIN_MS;
        xEventGroupSetBits(s_events, BIT_CONNECTED);
    }
}

void net_start(void) {
    s_events = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, on_event, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, on_event, NULL, NULL));

    wifi_config_t cfg = {0};
    strlcpy((char *)cfg.sta.ssid, WIFI_SSID, sizeof(cfg.sta.ssid));
    strlcpy((char *)cfg.sta.password, WIFI_PASSWORD, sizeof(cfg.sta.password));
    cfg.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &cfg));
    ESP_ERROR_CHECK(esp_wifi_start());
    // Modem sleep between our 60 s keepalives: a large part of the idle battery saving.
    esp_wifi_set_ps(WIFI_PS_MAX_MODEM);
}

bool net_wait_connected(int timeout_ms) {
    EventBits_t bits = xEventGroupWaitBits(s_events, BIT_CONNECTED, pdFALSE, pdTRUE, pdMS_TO_TICKS(timeout_ms));
    return (bits & BIT_CONNECTED) != 0;
}

bool net_is_connected(void) {
    return (xEventGroupGetBits(s_events) & BIT_CONNECTED) != 0;
}
