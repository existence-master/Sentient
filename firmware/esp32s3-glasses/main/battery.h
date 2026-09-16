// Battery percentage from an ADC divider (docs/NODES.md section 11.5).
#pragma once

void battery_init(void);
int battery_percent(void);  // 0..100, or -1 when the ADC is not available
int battery_millivolts(void);
