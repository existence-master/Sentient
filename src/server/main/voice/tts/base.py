from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Tuple, Union, Optional
import numpy as np

class TTSOptionsBase(dict): 
    pass

class BaseTTS(ABC):
    @abstractmethod
    async def stream_tts(self, text: str, language: Optional[str] = "en", options: TTSOptionsBase = None) -> AsyncGenerator[Union[bytes, Tuple[int, np.ndarray]], None]:
        """
        Streams synthesized audio for the given text.
        Yields audio chunks. Chunks can be bytes (preferred for direct network sending)
        or a tuple of (sample_rate, numpy_array_of_floats) for some internal TTS like Orpheus.
        """
        if False: 
            yield b""