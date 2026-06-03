from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class AssistantResponse:
    content: str
    model: str
    provider: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None


class BaseAssistant(ABC):
    def __init__(self, system_prompt: str = None):
        self.conversation_history: List[Message] = []
        self.system_prompt = system_prompt or "You are a helpful personal assistant."

    @abstractmethod
    def chat(self, user_message: str) -> AssistantResponse:
        ...

    def reset(self):
        self.conversation_history = []

    def get_history(self) -> List[Message]:
        return self.conversation_history.copy()
