"""Async OpenAI-compatible LLM client for chat and vision tasks.

Connects to a LiteLLM gateway (or any OpenAI-compatible endpoint) using
the ``openai`` Python SDK.  Three models are wired:

  - ``chat_model``   (DeepSeek-V3.2)      — text generation / reflection
  - ``vision_model`` (gemini-3-flash-preview)  — image → text description
  - ``embedding_model`` is handled in embeddings.py
"""

from __future__ import annotations

import base64
import logging

from openai import AsyncOpenAI

from aimemo.core.config import settings

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = settings.resolve_api_key()
        base_url = settings.resolve_base_url()
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        _client = AsyncOpenAI(**kwargs)
    return _client


def reset_client() -> None:
    """Reset the cached client (useful for tests)."""
    global _client
    _client = None


async def chat_completion(
    prompt: str,
    system: str = "You are a helpful AI assistant.",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    client = get_openai_client()
    resp = await client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


async def analyze_memory(content: str) -> dict:
    """Analyze content and return structured metadata for memory storage.

    Returns a dict with keys: memory_type, importance, tags.
    """
    client = get_openai_client()
    resp = await client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a memory classification engine. Analyze the given text and output "
                    "ONLY a JSON object with exactly these fields:\n"
                    '- "memory_type": one of "episodic", "semantic", "procedural"\n'
                    '  - episodic: events, experiences, conversations, things that happened\n'
                    '  - semantic: facts, knowledge, definitions, concepts, preferences\n'
                    '  - procedural: instructions, how-to, steps, workflows, rules\n'
                    '- "importance": float 0.0-1.0 based on:\n'
                    "  - 0.1-0.3: trivial/transient (greetings, small talk)\n"
                    "  - 0.4-0.6: normal (general info, routine events)\n"
                    "  - 0.7-0.8: significant (preferences, key facts, decisions)\n"
                    "  - 0.9-1.0: critical (core identity, safety rules, system constraints)\n"
                    '- "tags": list of 2-5 short keyword tags for retrieval, in the same language as the content\n'
                    "Output ONLY valid JSON, no markdown fences, no explanation."
                ),
            },
            {"role": "user", "content": content},
        ],
        temperature=0.2,
        max_tokens=256,
    )
    import json

    raw = resp.choices[0].message.content or "{}"
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


async def vision_describe(image_data: bytes, mime_type: str = "image/png", prompt: str = "") -> str:
    """Send an image to the vision model and get a text description."""
    client = get_openai_client()
    b64 = base64.b64encode(image_data).decode()
    data_url = f"data:{mime_type};base64,{b64}"

    user_text = prompt or "Describe this image in detail. Extract all text, objects, and context visible."

    resp = await client.chat.completions.create(
        model=settings.vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=1024,
    )
    return resp.choices[0].message.content or ""
