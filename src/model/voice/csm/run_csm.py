import os
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
from dataclasses import dataclass

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

# Default prompts are available at https://hf.co/sesame/csm-1b
prompt_filepath_conversational_a = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_a.wav"
)
prompt_filepath_conversational_b = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_b.wav"
)

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam I didn't do that "
            "for the exam we had recently to be fair that was a more of a last minute scenario "
            "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            "sort of start the day with this not like a panic but like a"
        ),
        "audio": prompt_filepath_conversational_a
    },
    "conversational_b": {
        "text": (
            "like a super Mario level. Like it's very like high detail. And like, once you get "
            "into the park, it just like, everything looks like a computer game and they have all "
            "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
            "will have like a question block. And if you like, you know, punch it, a coin will "
            "come out. So like everyone, when they come into the park, they get like this little "
            "bracelet and then you can go punching question blocks around."
        ),
        "audio": prompt_filepath_conversational_b
    }
}

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    # Resample is lazy so we can always call it
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

def main():
    # Select the best available device, skipping MPS due to float64 limitations
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load model
    generator = load_csm_1b(device)

    # Prepare prompts
    prompt_a = prepare_prompt(
        SPEAKER_PROMPTS["conversational_a"]["text"],
        0,
        SPEAKER_PROMPTS["conversational_a"]["audio"],
        generator.sample_rate
    )

    prompt_b = prepare_prompt(
        SPEAKER_PROMPTS["conversational_b"]["text"],
        1,
        SPEAKER_PROMPTS["conversational_b"]["audio"],
        generator.sample_rate
    )

    # Generate conversation
    conversation = [
        {"text": "Hey how are you doing?", "speaker_id": 0},
        {"text": "Pretty good, pretty good. How about you?", "speaker_id": 1},
        {"text": "I'm great! So happy to be speaking with you today.", "speaker_id": 0},
        {"text": "Me too! This is some cool stuff, isn't it?", "speaker_id": 1}
    ]

    # Generate each utterance
    generated_segments = []
    prompt_segments = [prompt_a, prompt_b]

    for utterance in conversation:
        print(f"Generating: {utterance['text']}")
        audio_tensor = generator.generate(
            text=utterance['text'],
            speaker=utterance['speaker_id'],
            context=prompt_segments + generated_segments,
            max_audio_length_ms=10_000,
        )
        generated_segments.append(Segment(text=utterance['text'], speaker=utterance['speaker_id'], audio=audio_tensor))

    # Concatenate all generations
    all_audio = torch.cat([seg.audio for seg in generated_segments], dim=0)
    torchaudio.save(
        "full_conversation.wav",
        all_audio.unsqueeze(0).cpu(),
        generator.sample_rate
    )
    print("Successfully generated full_conversation.wav")

if __name__ == "__main__":
    main() 