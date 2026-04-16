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
    For semantic memories also returns: subject, predicate, object_value, update_type.
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
                    '- "tags": list of 2-5 short keyword tags for retrieval, in the same language '
                    "as the content\n"
                    "\n"
                    'When memory_type is "semantic", ALSO include these fields:\n'
                    '- "subject": the entity the fact is about (string)\n'
                    '- "predicate": the relationship or attribute (string)\n'
                    '- "object_value": the value or target of the relationship (string)\n'
                    '- "update_type": one of "new_fact", "temporal_update", "contradiction", '
                    '"refinement", "duplicate"\n'
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


async def detect_contradiction(new_content: str, existing_contents: list[str]) -> dict:
    """Check if new_content contradicts any existing facts.

    Returns a dict with:
      - "has_contradiction": bool
      - "contradicted_indices": list[int] — indices into existing_contents
      - "merged_fact": str — updated fact incorporating the new information
    """
    if not existing_contents:
        return {"has_contradiction": False, "contradicted_indices": [], "merged_fact": ""}

    client = get_openai_client()
    numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(existing_contents))
    resp = await client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a fact-consistency checker. Given a NEW fact and a list of EXISTING facts, "
                    "determine if the new fact contradicts, updates, or supersedes any existing fact.\n"
                    "Output ONLY a JSON object:\n"
                    '- "has_contradiction": true if the new fact conflicts with or updates any existing fact\n'
                    '- "contradicted_indices": list of integer indices [i] of the contradicted existing facts\n'
                    '- "merged_fact": if contradicted, a single concise sentence merging the latest truth; '
                    "empty string if no contradiction\n"
                    "Contradiction includes: changed values, updated preferences, corrected info, "
                    "reversed decisions. NOT contradiction: additional detail on same topic.\n"
                    "Output ONLY valid JSON, no markdown fences."
                ),
            },
            {
                "role": "user",
                "content": f"NEW FACT:\n{new_content}\n\nEXISTING FACTS:\n{numbered}",
            },
        ],
        temperature=0.1,
        max_tokens=512,
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


async def analyze_emotion_signals(raw_text: str) -> dict:
    """Analyze emotional signals from one user utterance.

    Returns JSON-compatible keys:
    intent, support_need, vulnerability_signal, attachment_signal, conflict_signal
    """
    client = get_openai_client()
    resp = await client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an emotion signal extractor for conversational support systems. "
                    "Output ONLY valid JSON with these fields:\n"
                    '- "intent": one of "chat", "support", "advice", "conflict"\n'
                    '- "support_need": float 0.0-1.0\n'
                    '- "vulnerability_signal": float 0.0-1.0\n'
                    '- "attachment_signal": float 0.0-1.0\n'
                    '- "conflict_signal": float 0.0-1.0\n'
                    "No markdown, no explanation."
                ),
            },
            {"role": "user", "content": raw_text},
        ],
        temperature=0.1,
        max_tokens=180,
    )
    import json

    raw = resp.choices[0].message.content or "{}"
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(raw)
