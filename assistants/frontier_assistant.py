import time
import os
from typing import Optional

import anthropic

from .base import BaseAssistant, Message, AssistantResponse


class FrontierAssistant(BaseAssistant):
    """Frontier assistant using Claude Sonnet 4.6 via Anthropic API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_context_messages: int = 20,
    ):
        super().__init__(system_prompt=system_prompt)
        self.model = "claude-sonnet-4-6"
        self.provider = "anthropic"
        self.max_context_messages = max_context_messages

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=key)

    def chat(self, user_message: str) -> AssistantResponse:
        """Send a message and return the assistant's response."""
        self.conversation_history.append(Message(role="user", content=user_message))

        # Keep only the last N messages to avoid token limits
        recent_history = self.conversation_history[-self.max_context_messages :]

        # Build messages list for the API (no system message in messages array)
        messages = [
            {"role": msg.role, "content": msg.content} for msg in recent_history
        ]

        start_time = time.monotonic()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.system_prompt,
                messages=messages,
            )
            latency_ms = (time.monotonic() - start_time) * 1000

            content = response.content[0].text if response.content else ""
            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0

            self.conversation_history.append(
                Message(role="assistant", content=content)
            )

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
            self.conversation_history.append(
                Message(role="assistant", content=error_msg)
            )
            return AssistantResponse(
                content=error_msg,
                model=self.model,
                provider=self.provider,
                latency_ms=round(latency_ms, 2),
                error=str(e),
            )
