"""
OpenAI-compatible response dataclasses.

Non-OpenAI providers (Anthropic, Gemini) convert their native responses into
these dataclasses so the Agent can use a single code path regardless of which
LLM backend is active.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class MockFunction:
    name: str
    arguments: str


@dataclass
class MockToolCall:
    id: str
    function: MockFunction
    type: str = "function"


@dataclass
class MockMessage:
    content: Optional[str]
    tool_calls: Optional[list[MockToolCall]]
    reasoning_content: Optional[str] = None

    def model_dump(self) -> dict:
        d: dict = {"role": "assistant", "content": self.content}
        if self.reasoning_content:
            # Internal-only context required by DeepSeek when continuing a
            # thinking-mode tool cycle. This is never streamed to the user.
            d["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return d


@dataclass
class MockChoice:
    message: MockMessage


@dataclass
class MockResponse:
    choices: list[MockChoice]
