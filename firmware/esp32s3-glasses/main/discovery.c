#include "discovery.h"

#include <string.h>

#include "app_config.h"
#include "esp_log.h"
#include "mdns.h"

static const char *TAG = "discovery";

void discovery_init(void) {
    ESP_ERROR_CHECK(mdns_init());
    mdns_hostname_set("sentient-glasses");
    mdns_instance_name_set(NODE_NAME);
}

static void copy_txt(mdns_result_t *r, const char *key, char *out, size_t cap) {
    out[0] = '\0';
    for (size_t i = 0; i < r->txt_count; i++) {
        if (r->txt[i].key && strcmp(r->txt[i].key, key) == 0 && r->txt[i].value) {
            strlcpy(out, r->txt[i].value, cap);
            return;
        }
    }
}

bool discovery_find(engine_addr_t *out, int timeout_ms) {
    memset(out, 0, sizeof(*out));
    mdns_result_t *results = NULL;
    // "_sentient" / "_tcp": at most 4 answers, we take the first one with an IPv4 address.
    esp_err_t err = mdns_query_ptr("_sentient", "_tcp", timeout_ms, 4, &results);
    if (err != ESP_OK || results == NULL) {
        ESP_LOGW(TAG, "no Sentient engine answered on this network");
        if (results) mdns_query_results_free(results);
        return false;
    }
    bool found = false;
    for (mdns_result_t *r = results; r != NULL && !found; r = r->next) {
        for (mdns_ip_addr_t *a = r->addr; a != NULL; a = a->next) {
            if (a->addr.type != ESP_IPADDR_TYPE_V4) continue;
            snprintf(out->host, sizeof(out->host), IPSTR, IP2STR(&a->addr.u_addr.ip4));
            out->port = r->port ? r->port : ENGINE_PORT;
            copy_txt(r, "path", out->path, sizeof(out->path));
            copy_txt(r, "fingerprint", out->fingerprint, sizeof(out->fingerprint));
            if (out->path[0] == '\0') strlcpy(out->path, "/ws/node", sizeof(out->path));
            ESP_LOGI(TAG, "engine at %s:%d%s", out->host, out->port, out->path);
            found = true;
            break;
        }
    }
    mdns_query_results_free(results);
    return found;
}
