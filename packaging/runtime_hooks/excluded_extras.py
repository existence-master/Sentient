"""Turn "this extra was left out of the installer" into a sentence a person can act on.

The installer deliberately does not bundle the heavy optional extras (local
speech recognition, neural TTS, the wake-word detector, OpenCV). Code that
reaches for them already guards the import, but the raw
``No module named 'faster_whisper'`` that a guard re-raises is meaningless to a
non-technical user. This runtime hook installs a meta-path finder that raises
the same ``ModuleNotFoundError`` (so every ``except ImportError`` keeps working)
with a message that says what is missing and what to do instead.

Kept in `packaging/` on purpose: it only exists for frozen builds and must not
change how the engine behaves when run from source.
"""

from __future__ import annotations

import sys
from importlib.abc import MetaPathFinder

_MESSAGES = {
    "faster_whisper": (
        "Speaking to Sentient needs the optional speech pack, which isn't part of this "
        "installer. Open Settings -> Voice and choose a cloud speech provider (OpenAI, "
        "Deepgram or ElevenLabs), or install Sentient from source with the 'voice' extra."
    ),
    "ctranslate2": (
        "The optional speech pack isn't part of this installer. Choose a cloud speech "
        "provider in Settings -> Voice, or install Sentient from source with the 'voice' extra."
    ),
    "kokoro_onnx": (
        "The Kokoro voice isn't part of this installer. Open Settings -> Voice and pick the "
        "system voice or a cloud voice instead."
    ),
    "openwakeword": (
        "The neural wake-word detector isn't part of this installer. Open Settings -> Voice "
        "and use the Whisper wake word, or turn the wake word off."
    ),
    "onnxruntime": (
        "This feature needs the optional on-device model runtime, which isn't part of this "
        "installer. Use a cloud provider for it in Settings instead."
    ),
    "cv2": (
        "Webcam capture through OpenCV isn't part of this installer. Use the phone or glasses "
        "device app for the camera instead."
    ),
    "torch": (
        "PyTorch isn't part of this installer. Use a cloud provider for this feature in Settings."
    ),
}


class _ExcludedExtrasFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        """`path`/`target` are unused: this finder never resolves, it only explains."""
        root = fullname.partition(".")[0]
        if root in _MESSAGES:
            raise ModuleNotFoundError(_MESSAGES[root], name=fullname)
        return  # None: let the next finder try, so a genuine miss raises the usual error


if getattr(sys, "frozen", False):
    sys.meta_path.append(_ExcludedExtrasFinder())
