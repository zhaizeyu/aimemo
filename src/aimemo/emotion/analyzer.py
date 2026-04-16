"""Emotion signal extraction.

LLM classification can be plugged in later, but state updates remain policy-driven.
"""

from __future__ import annotations

import re

from aimemo.core.config import settings
from aimemo.core.llm import analyze_emotion_signals
from aimemo.emotion.models import EmotionEvent

_NEG_WORDS = ("难过", "焦虑", "崩溃", "痛苦", "糟糕", "sad", "anxious", "stress", "撑不住", "顶不住")
_POS_WORDS = ("开心", "高兴", "感谢", "谢谢", "棒", "great", "love")
_URGENT_WORDS = ("马上", "立刻", "紧急", "救命", "urgent", "asap")
_ADVICE_WORDS = ("怎么办", "建议", "plan", "advice", "应该")
_SUPPORT_WORDS = ("陪", "听我", "倾诉", "安慰", "抱抱", "help me")
_CONFLICT_WORDS = ("闭嘴", "滚", "讨厌", "你错", "hate", "stupid")
_ATTACHMENT_WORDS = ("别离开", "需要你", "想你", "陪着我", "miss you")
_PLAYFUL_WORDS = ("哈哈", "lol", "玩笑", "逗", "😄")
_HIDDEN_SUPPORT_CUES = (
    "其实",
    "有点扛不住",
    "撑不住",
    "顶不住",
    "不知道和谁说",
    "睡不着",
    "没人懂",
    "心里很乱",
)
_RESTRAINED_VULNERABILITY_CUES = (
    "没事",
    "不用担心",
    "还好",
    "一般般",
    "就那样",
    "算了",
)


class EmotionAnalyzer:
    """Rule-first analyzer; suitable for deterministic tests and explainability."""

    def __init__(self, use_llm_assist: bool | None = None):
        if use_llm_assist is None:
            use_llm_assist = bool(settings.resolve_api_key() and settings.resolve_base_url())
        self.use_llm_assist = use_llm_assist

    async def analyze(self, *, raw_text: str, user_id: str, session_id: str | None = None) -> EmotionEvent:
        text = raw_text.lower()

        sentiment = self._score_sentiment(text)
        support_need = self._contains_any(text, _SUPPORT_WORDS)
        advice_signal = self._contains_any(text, _ADVICE_WORDS)
        hidden_support = self._contains_any(text, _HIDDEN_SUPPORT_CUES)
        restrained_vulnerability = self._contains_any(text, _RESTRAINED_VULNERABILITY_CUES)
        vulnerability = self._detect_vulnerability(text, sentiment, hidden_support, restrained_vulnerability)
        intent = self._infer_intent(
            advice_signal=advice_signal,
            support_need=support_need,
            vulnerability=vulnerability,
            hidden_support=hidden_support,
        )

        if (support_need > 0.45 or hidden_support > 0.35) and intent == "chat":
            intent = "support"

        event = EmotionEvent(
            raw_text=raw_text,
            user_id=user_id,
            session_id=session_id,
            intent=intent,
            sentiment=sentiment,
            urgency=self._contains_any(text, _URGENT_WORDS),
            support_need=max(support_need, hidden_support, vulnerability * 0.75, 0.7 if sentiment < -0.4 else 0.0),
            attachment_signal=self._contains_any(text, _ATTACHMENT_WORDS),
            praise_signal=1.0 if re.search(r"谢谢|感谢|太好了|awesome|great", text) else 0.0,
            conflict_signal=self._contains_any(text, _CONFLICT_WORDS),
            vulnerability_signal=vulnerability,
            playfulness_signal=self._contains_any(text, _PLAYFUL_WORDS),
            confidence=0.75 if hidden_support == 0 else 0.72,
        )
        if not self.use_llm_assist:
            return event

        try:
            llm_signals = await analyze_emotion_signals(raw_text)
        except Exception:
            return event

        llm_intent = str(llm_signals.get("intent", "")).strip().lower()
        if llm_intent in {"chat", "support", "advice", "conflict"}:
            event.intent = llm_intent
        event.support_need = self._blend(event.support_need, llm_signals.get("support_need"))
        event.vulnerability_signal = self._blend(
            event.vulnerability_signal, llm_signals.get("vulnerability_signal")
        )
        event.support_need = self._blend(event.support_need, llm_signals.get("empathy_need"))
        event.attachment_signal = self._blend(event.attachment_signal, llm_signals.get("attachment_signal"))
        event.conflict_signal = self._blend(event.conflict_signal, llm_signals.get("conflict_signal"))
        event.confidence = 0.85
        return event

    @staticmethod
    def _detect_vulnerability(
        text: str, sentiment: float, hidden_support: float, restrained_vulnerability: float
    ) -> float:
        explicit = 1.0 if re.search(r"害怕|不安|脆弱|孤单|委屈|i feel alone|i'm not okay", text) else 0.0
        muted = 0.65 if restrained_vulnerability > 0 and sentiment < 0 else 0.0
        return max(explicit, hidden_support * 0.8, muted)

    @staticmethod
    def _infer_intent(*, advice_signal: float, support_need: float, vulnerability: float, hidden_support: float) -> str:
        # "表面求建议，实际求共情": vulnerable + advice defaults to support.
        if advice_signal > 0 and (vulnerability > 0.35 or hidden_support > 0.3):
            return "support"
        if advice_signal > 0:
            return "advice"
        if support_need > 0.5 or vulnerability > 0.45:
            return "support"
        return "chat"

    def _score_sentiment(self, text: str) -> float:
        pos = self._contains_any(text, _POS_WORDS)
        neg = self._contains_any(text, _NEG_WORDS)
        return max(-1.0, min(1.0, pos - neg))

    @staticmethod
    def _blend(rule_score: float, llm_score: object) -> float:
        try:
            val = float(llm_score)
        except (TypeError, ValueError):
            return rule_score
        val = max(0.0, min(1.0, val))
        return max(0.0, min(1.0, 0.6 * rule_score + 0.4 * val))

    @staticmethod
    def _contains_any(text: str, keywords: tuple[str, ...]) -> float:
        hits = sum(1 for w in keywords if w in text)
        if hits == 0:
            return 0.0
        return min(1.0, hits / max(1, len(keywords) / 2))
