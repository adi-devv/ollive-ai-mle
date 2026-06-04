import time
import os
from typing import Optional

from groq import Groq

from .base import BaseAssistant, Message, AssistantResponse


class FrontierAssistant(BaseAssistant):
    """Frontier assistant using Llama 3.3 70B via Groq API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        system_prompt: Optional[str] = None,
        max_context_messages: int = 20,
    ):
        super().__init__(system_prompt=system_prompt)
        self.model = model
        self.provider = "groq"
        self.max_context_messages = max_context_messages

        key = api_key or os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=key)

    def chat(self, user_message: str) -> AssistantResponse:
        """Send a message and return the assistant's response."""
        self.conversation_history.append(Message(role="user", content=user_message))

        # System prompt + last N messages
        messages = [{"role": "system", "content": self.system_prompt}]
        recent_history = self.conversation_history[-self.max_context_messages:]
        for msg in recent_history:
            messages.append({"role": msg.role, "content": msg.content})

        start_time = time.monotonic()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                messages=messages,
            )
            latency_ms = (time.monotonic() - start_time) * 1000

            content = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            self.conversation_history.append(Message(role="assistant", content=content))

            return AssistantResponse(
                content=content,
                model=self.model,
                provider=self.provider,
                latency_ms=round(latency_ms, 2),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            error_msg = f"[Error: {type(e).__name__}: {e}]"
            self.conversation_history.append(Message(role="assistant", content=error_msg))
            return AssistantResponse(
                content=error_msg,
                model=self.model,
                provider=self.provider,
                latency_ms=round(latency_ms, 2),
                error=str(e),
            )
