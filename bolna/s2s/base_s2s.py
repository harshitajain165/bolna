from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional


class BaseS2SProvider(ABC):
    """Base class for speech-to-speech providers.

    Any S2S provider (OpenAI Realtime, Gemini Live, etc.) implements this
    interface.  TaskManager only interacts through these methods, keeping
    the provider details isolated.
    """

    def __init__(
        self,
        *,
        system_prompt: str,
        voice: str,
        model: str,
        api_key: str,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ):
        self.system_prompt = system_prompt
        self.voice = voice
        self.model = model
        self.api_key = api_key
        self.tools = tools or []
        self.connection_time: Optional[float] = None
        self.turn_latencies: list = []

    @abstractmethod
    async def connect(self) -> None:
        """Open the WebSocket / session to the provider."""
        ...

    @abstractmethod
    async def send_audio(self, pcm_24k_bytes: bytes) -> None:
        """Send PCM-16 24 kHz mono audio to the provider."""
        ...

    @abstractmethod
    def receive_events(self) -> AsyncGenerator:
        """Yield provider-agnostic S2S events (AudioDelta, TranscriptDelta, etc.).

        Implementations are async generators (``async def`` with ``yield``).
        Declared here as a regular abstract method so the base class isn't itself
        a generator stub.
        """

    @abstractmethod
    async def send_function_result(self, call_id: str, result: str) -> None:
        """Return the result of a function call back to the provider."""
        ...

    @abstractmethod
    async def commit_function_results(self) -> None:
        """Tell the provider to continue after function_call_output(s) submitted."""
        ...

    @abstractmethod
    async def trigger_response(self, instructions: Optional[str] = None) -> None:
        """Ask the provider to generate a response (e.g. welcome message)."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Cleanly close the connection."""
        ...
