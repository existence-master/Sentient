idf_component_register(
    SRCS
        "main.c"
        "net.c"
        "storage.c"
        "discovery.c"
        "cert_pin.c"
        "node_client.c"
        "caps.c"
        "audio.c"
        "camera.c"
        "voice.c"
        "button.c"
        "battery.c"
    INCLUDE_DIRS "."
    REQUIRES
        nvs_flash esp_wifi esp_event esp_netif esp_timer esp-tls mbedtls json driver esp_adc
)
