from sentient.voice.audio import pcm16_to_float, pcm16_to_wav, resample, wav_to_pcm16
from sentient.voice.session import parse_decision
from sentient.voice.text import SentenceSplitter, clean_for_speech, split_sentences
from sentient.voice.vad import EnergyVAD
from tests.voice.conftest import frames, noise, silence, tone


# ----------------------------------------------------------------------------- sentence splitting
def test_basic_sentences():
    assert split_sentences("Hello there. How can I help?") == ["Hello there.", "How can I help?"]


def test_abbreviations_numbers_initials():
    text = "Dr. Smith paid $3.50 at 9 a.m. today. J. K. Rowling wrote it, e.g. in 1997. Meet at No. 5 now."
    assert split_sentences(text) == [
        "Dr. Smith paid $3.50 at 9 a.m. today.",
        "J. K. Rowling wrote it, e.g. in 1997.",
        "Meet at No. 5 now.",
    ]


def test_real_sentence_ending_in_no_and_number():
    assert split_sentences("I said no. Then I left in 2024. Done!") == ["I said no.", "Then I left in 2024.", "Done!"]


def test_streaming_emits_as_soon_as_provable():
    s = SentenceSplitter()
    out: list[str] = []
    stream = "Sure. The total is 3.5 dollars. Next one"
    emitted_at = {}
    for i, ch in enumerate(stream):
        new = s.feed(ch)
        for sentence in new:
            emitted_at[sentence] = i
        out += new
    assert out == ["Sure.", "The total is 3.5 dollars."]
    assert emitted_at["Sure."] == len("Sure. ") - 1  # right after the space arrives
    assert s.flush() == ["Next one"]


def test_decimal_split_across_deltas_is_not_a_boundary():
    s = SentenceSplitter()
    assert s.feed("It costs 3.") == []
    assert s.feed("50 today. ") == ["It costs 3.50 today."]


def test_lists_newlines_and_code_blocks():
    text = "Here is the plan:\n1. Buy milk\n2. Call mom\n```python\nprint('x. y. z.')\n```\nAll done."
    assert split_sentences(text) == ["Here is the plan:", "Buy milk", "Call mom", "All done."]


def test_code_fence_split_across_deltas():
    s = SentenceSplitter()
    out = []
    for chunk in ["Run this: `", "``bash\nls -la. rm x.", " done\n``", "` Then relax. "]:
        out += s.feed(chunk)
    out += s.flush()
    assert out == ["Run this:", "Then relax."]


def test_long_run_on_sentence_is_soft_split():
    s = SentenceSplitter(max_chars=60)
    out = s.feed("word, " * 30)
    assert out and all(len(x) <= 60 for x in out)


def test_clean_for_speech_markdown_urls_symbols():
    md = "## Weather\n- **Sunny** and _warm_ in `Pune` ~~maybe~~ 🌞\nSee [the forecast](https://example.com/x) or https://foo.bar/baz."
    assert clean_for_speech(md) == "Weather Sunny and warm in Pune maybe See the forecast or."
    assert clean_for_speech("| a | b |\n|---|---|\n| 1 | 2 |") == "a, b, 1, 2"
    assert clean_for_speech("Tom & Jerry <b>rock</b>") == "Tom and Jerry rock"
    assert split_sentences("```\nonly code\n```") == []


def test_parse_decision():
    assert parse_decision("Yes, go ahead.") == "allow"
    assert parse_decision("okay") == "allow"
    assert parse_decision("No, don't do that") == "deny"
    assert parse_decision("What's the weather?") is None


# ----------------------------------------------------------------------------- audio helpers
def test_wav_roundtrip_and_resample():
    pcm = tone(100, sr=22050)
    wav = pcm16_to_wav(pcm, 22050)
    back, sr = wav_to_pcm16(wav)
    assert sr == 22050 and back == pcm
    assert resample(pcm16_to_float(pcm), 22050, 16000).size == 1600
    assert resample(pcm16_to_float(tone(100, sr=48000)), 48000, 16000).size == 1600


# ----------------------------------------------------------------------------- VAD
def _run(vad: EnergyVAD, pcm: bytes):
    events = []
    for f in frames(pcm):
        events += vad.feed(f)
    return events


def test_vad_silence_only():
    assert _run(EnergyVAD(16000), silence(2000)) == []


def test_vad_tone_then_silence_is_one_utterance():
    vad = EnergyVAD(16000, silence_ms=500)
    events = _run(vad, silence(300) + tone(800) + silence(700))
    kinds = [e.kind for e in events]
    assert kinds == ["speech_start", "utterance"]
    utt = events[1]
    assert utt.reason == "silence"
    # includes the tone, a little pre-roll and ~200 ms of tail, but not the full silence
    assert 800 <= utt.duration_ms <= 800 + 300 + 260


def test_vad_short_blip_is_ignored():
    events = _run(EnergyVAD(16000, silence_ms=300, min_speech_ms=250), tone(120) + silence(600))
    assert [e.kind for e in events] == ["speech_start"]


def test_vad_two_utterances_and_irregular_chunks():
    vad = EnergyVAD(16000, silence_ms=400)
    pcm = tone(500) + silence(600) + tone(400, freq=330) + silence(600)
    events = []
    i = 0
    for size in [123, 999, 4096, 7, 20000] * 20:  # odd chunk sizes, including odd byte counts
        if i >= len(pcm):
            break
        events += vad.feed(pcm[i : i + size])
        i += size
    assert [e.kind for e in events].count("utterance") == 2


def test_vad_max_length_forces_end():
    events = _run(EnergyVAD(16000, max_utterance_ms=1000), tone(2500))
    utts = [e for e in events if e.kind == "utterance"]
    assert utts and utts[0].reason == "max_length" and utts[0].duration_ms <= 1020


def test_vad_adapts_to_background_noise():
    vad = EnergyVAD(16000, silence_ms=400)
    assert _run(vad, noise(1500, amp=0.008)) == []
    events = _run(vad, tone(600) + noise(800, amp=0.008, seed=1))
    assert [e.kind for e in events] == ["speech_start", "utterance"]


def test_vad_push_to_talk_flush():
    vad = EnergyVAD(16000, silence_ms=3000)
    _run(vad, tone(600))
    assert vad.in_speech
    ev = vad.flush()
    assert ev is not None and ev.reason == "manual" and ev.duration_ms >= 600
    assert vad.flush() is None
