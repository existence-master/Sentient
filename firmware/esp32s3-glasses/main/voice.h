// Hold-to-talk over WS /ws/voice?node_token= (docs/API.md section 9, docs/NODES.md section 6).
#pragma once
#include <stdbool.h>

void voice_init(const char *host, int port, const char *cert_pem);

void voice_press(void);   // open the socket if needed and start streaming the microphone
void voice_release(void); // end the utterance, keep the socket open for the reply
void voice_interrupt(void);
bool voice_is_speaking(void);
