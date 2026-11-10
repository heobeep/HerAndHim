"""Regression tests for the OpenAI-compatible provider's reasoning handling."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from herandhim.core.llm.openai_compatible import OpenAICompatibleProvider


def _provider(model_name: str = "deepseek-v4.1-flash:free") -> OpenAICompatibleProvider:
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.model_name = model_name
    provider.supports_images = False
    provider.client = MagicMock()
    return provider


def _chunk(*, content=None, reasoning_content=None):
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _collect_generator(generator):
    events = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as stop:
            return events, stop.value


def test_chat_never_promotes_reasoning_to_visible_content():
    provider = _provider()
    message = SimpleNamespace(
        content=None,
        reasoning_content="This is private reasoning.",
        tool_calls=None,
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    provider.client.chat.completions.create.return_value = response

    result = provider.chat([{"role": "user", "content": "Hi"}])

    assert result.choices[0].message.content is None
    _, kwargs = provider.client.chat.completions.create.call_args
    assert kwargs["extra_body"]["thinking"]["type"] == "enabled"


def test_stream_ignores_reasoning_and_only_yields_final_content():
    provider = _provider()
    provider.client.chat.completions.create.return_value = iter([
        _chunk(reasoning_content="Private internal reasoning."),
        _chunk(content="Hey! Good to see you."),
    ])

    events, response = _collect_generator(
        provider.chat_stream([{"role": "user", "content": "Hi"}])
    )

    assert events == [{"type": "text_delta", "text": "Hey! Good to see you."}]
    assert response.choices[0].message.content == "Hey! Good to see you."
    assert response.choices[0].message.reasoning_content == "Private internal reasoning."
    assert response.choices[0].message.model_dump()["reasoning_content"] == "Private internal reasoning."
    _, kwargs = provider.client.chat.completions.create.call_args
    assert kwargs["stream"] is True
    assert kwargs["extra_body"]["thinking"]["type"] == "enabled"


def test_non_deepseek_model_does_not_get_deepseek_thinking_options():
    provider = _provider("gpt-4o-mini")
    provider.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello"))]
    )

    provider.chat([{"role": "user", "content": "Hi"}])

    _, kwargs = provider.client.chat.completions.create.call_args
    assert "extra_body" not in kwargs
