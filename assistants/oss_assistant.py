import time
import os
from typing import Optional

from huggingface_hub import InferenceClient

from .base import BaseAssistant, Message, AssistantResponse


class OSSAssistant(BaseAssistant):
    """Open-source assistant using Qwen 2.5 via HuggingFace Inference API."""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        system_prompt: Optional[str] = None,
        max_context_messages: int = 20,
    ):
        super().__init__(system_prompt=system_prompt)
        self.model = model
        self.provider = "huggingface"
        self.max_context_messages = max_context_messages

        token = hf_token or os.environ.get("HF_TOKEN")
        self.client = InferenceClient(model=model, token=token)

    def chat(self, user_message: str) -> AssistantResponse:
        """Send a message and return the assistant's response."""
        self.conversation_history.append(Message(role="user", content=user_message))

        # Build messages list: system prompt + recent history
        messages = [{"role": "system", "content": self.system_prompt}]

        # Keep only the last N messages to avoid token limits
        recent_history = self.conversation_history[-self.max_context_messages :]
        for msg in recent_history:
            messages.append({"role": msg.role, "content": msg.content})

        start_time = time.monotonic()
        try:
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.7,
            )
            latency_ms = (time.monotonic() - start_time) * 1000

            content = response.choices[0].message.content or ""

            # Extract token usage if available
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage") and response.usage:
                input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(response.usage, "completion_tokens", 0) or 0

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
