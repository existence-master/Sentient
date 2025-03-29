## Outetts

import outetts
import time

# --- Start timing model loading ---
start_load_time = time.perf_counter()
print("Configuring model...")

model_config = outetts.GGUFModelConfig_v1(
    model_path="models/OuteTTS-0.2-500M-f16.gguf",
    language="en", # Supported languages in v0.2: en, zh, ja, ko
    n_gpu_layers=-1,
)

print("Initializing interface and loading model...")
interface = outetts.InterfaceGGUF(model_version="0.2", cfg=model_config)

# --- End timing model loading ---
end_load_time = time.perf_counter()

# Calculate and print the model loading time
model_load_time = end_load_time - start_load_time
print(f"Model loading finished.")
print(f"Model loading took: {model_load_time:.3f} seconds") # Print duration formatted to 3 decimal places


# Print available default speakers
interface.print_default_speakers()

# Load a default speaker
print("Loading default speaker...")
# Note: Speaker loading might involve minimal disk I/O but is usually very fast
# compared to model loading. We are not timing this separately here,
# but you could add timers around this call if needed.
speaker = interface.load_default_speaker(name="male_1")
print("Speaker loaded.")


# --- Start timing the generation process ---
start_gen_time = time.perf_counter()
print("Starting audio generation...")

# Generate speech
output = interface.generate(
    text="Speech synthesis is the artificial production of human speech.",
    temperature=0.7,
    repetition_penalty=1.1,
    max_length=4096,

    # Optional: Use a speaker profile for consistent voice characteristics
    # Without a speaker profile, the model will generate a voice with random characteristics
    speaker=speaker,
)

# --- End timing the generation process ---
end_gen_time = time.perf_counter()

# Calculate and print the generation time
generation_time = end_gen_time - start_gen_time
print(f"Audio generation finished.")
print(f"Audio generation took: {generation_time:.3f} seconds") # Print duration formatted to 3 decimal places

# Save the generated speech to a file
print("Saving audio to output.wav...")
output.save("output.wav")
print("Audio saved successfully to output.wav")