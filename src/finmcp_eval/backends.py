"""Chat backends: Gemini (default) and a local Ollama fallback.

Each backend translates between its own message format and the small neutral
types below. Tools are passed only in the with-tools mode; everything else
(model, system instruction, generation settings) is identical in both modes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .config import GenerationSettings
from .schemas import ToolSpec

log = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    args: Any
    id: str | None = None


@dataclass
class ModelTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class BackendError(RuntimeError):
    """A model API failure. Messages contain no secrets."""


class ChatBackend(Protocol):
    """`start` returns opaque per-question state that the other two methods update."""

    name: str
    model: str

    def start(self, system_instruction: str, question: str, tools: list[ToolSpec] | None) -> Any: ...

    async def generate(self, conv: Any) -> ModelTurn: ...

    def add_tool_results(self, conv: Any, results: list[tuple[ToolCall, dict[str, Any]]]) -> None: ...


class _Pacer:
    """Keeps a minimum interval between requests (free-tier per-minute limits)."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last = float("-inf")

    async def wait(self) -> None:
        delay = self._last + self.min_interval - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        self._last = time.monotonic()


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


@dataclass
class _GeminiConv:
    contents: list[Any]
    config: Any


class GeminiBackend:
    name = "gemini"
    RETRY_CODES = frozenset({429, 500, 503})
    MAX_RETRIES = 3

    def __init__(
        self,
        model: str,
        settings: GenerationSettings,
        *,
        api_key: str | None = None,
        client: Any = None,
        min_interval: float = 0.0,
    ) -> None:
        from google import genai

        self.model = model
        self.settings = settings
        self._client = client if client is not None else genai.Client(api_key=api_key)
        self._pacer = _Pacer(min_interval)

    def start(self, system_instruction: str, question: str, tools: list[ToolSpec] | None) -> _GeminiConv:
        from google.genai import types

        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t.name, description=t.description, parameters_json_schema=t.json_schema()
                        )
                        for t in tools
                    ]
                )
            ]
            kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="AUTO", allowed_function_names=[t.name for t in tools]
                )
            )
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.settings.temperature,
            max_output_tokens=self.settings.max_output_tokens,
            seed=self.settings.seed,
            # We run tools ourselves through the MCP adapter; never let the SDK call Python functions.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            **kwargs,
        )
        contents = [types.Content(role="user", parts=[types.Part(text=question)])]
        return _GeminiConv(contents=contents, config=config)

    async def generate(self, conv: _GeminiConv) -> ModelTurn:
        from google.genai import errors

        for attempt in range(self.MAX_RETRIES + 1):
            await self._pacer.wait()
            try:
                response = await self._client.aio.models.generate_content(
                    model=self.model, contents=conv.contents, config=conv.config
                )
                break
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                if code in self.RETRY_CODES and attempt < self.MAX_RETRIES:
                    delay = min(60.0, 10.0 * 2**attempt)
                    log.warning("Gemini returned %s; retrying in %.0fs", code, delay)
                    await asyncio.sleep(delay)
                    continue
                detail = str(getattr(exc, "message", "") or "")[:300]
                raise BackendError(f"Gemini API error {code} {getattr(exc, 'status', '')}: {detail}") from None

        candidates = response.candidates or []
        if not candidates or candidates[0].content is None:
            reason = getattr(candidates[0], "finish_reason", None) if candidates else None
            raise BackendError(f"Gemini returned no content (finish_reason={reason})")
        content = candidates[0].content
        # Keep the model's turn verbatim (including thought signatures) for the next request.
        conv.contents.append(content)

        text_parts, calls = [], []
        for part in content.parts or []:
            if part.function_call is not None:
                fc = part.function_call
                calls.append(ToolCall(name=fc.name or "", args=dict(fc.args or {}), id=fc.id))
            elif part.text and not part.thought:
                text_parts.append(part.text)
        return ModelTurn(text="".join(text_parts).strip(), tool_calls=calls)

    def add_tool_results(self, conv: _GeminiConv, results: list[tuple[ToolCall, dict[str, Any]]]) -> None:
        from google.genai import types

        parts = [
            types.Part(function_response=types.FunctionResponse(id=call.id, name=call.name, response=payload))
            for call, payload in results
        ]
        conv.contents.append(types.Content(role="user", parts=parts))


# ---------------------------------------------------------------------------
# Ollama (optional local fallback)
# ---------------------------------------------------------------------------


@dataclass
class _OllamaConv:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None


class OllamaBackend:
    name = "ollama"

    def __init__(
        self,
        model: str,
        settings: GenerationSettings,
        *,
        host: str,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.settings = settings
        self._http = httpx.AsyncClient(
            base_url=host, timeout=timeout, follow_redirects=False, transport=transport
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    def start(self, system_instruction: str, question: str, tools: list[ToolSpec] | None) -> _OllamaConv:
        decls = (
            [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.json_schema()},
                }
                for t in tools
            ]
            if tools
            else None
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": question},
        ]
        return _OllamaConv(messages=messages, tools=decls)

    async def generate(self, conv: _OllamaConv) -> ModelTurn:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": conv.messages,
            "stream": False,
            "options": {
                "temperature": self.settings.temperature,
                "seed": self.settings.seed,
                "num_predict": self.settings.max_output_tokens,
            },
        }
        if conv.tools:
            body["tools"] = conv.tools
        try:
            resp = await self._http.post("/api/chat", json=body)
        except httpx.HTTPError as exc:
            raise BackendError(f"could not reach Ollama: {type(exc).__name__}") from None
        if resp.status_code != 200:
            raise BackendError(f"Ollama returned HTTP {resp.status_code}")
        try:
            message = resp.json()["message"]
        except (ValueError, KeyError, TypeError):
            raise BackendError("unexpected Ollama response format") from None
        conv.messages.append(message)

        calls = []
        for i, tc in enumerate(message.get("tool_calls") or []):
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass  # Left as a string; the adapter will reject it.
            calls.append(ToolCall(name=str(fn.get("name", "")), args=args, id=tc.get("id") or f"call_{i}"))
        return ModelTurn(text=str(message.get("content") or "").strip(), tool_calls=calls)

    def add_tool_results(self, conv: _OllamaConv, results: list[tuple[ToolCall, dict[str, Any]]]) -> None:
        for call, payload in results:
            conv.messages.append({"role": "tool", "tool_name": call.name, "content": json.dumps(payload)})
