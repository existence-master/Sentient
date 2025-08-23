from abc import ABC, abstractmethod
from typing import Tuple, Optional

class BaseSTT(ABC):
    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, sample_rate: int) -> Tuple[str, Optional[str]]:
        """
        Transcribes audio bytes to text.
        Returns a tuple of (transcription, detected_language_code).
        Language code should be BCP-47.
        """
        pass