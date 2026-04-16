from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aimemo.emotion.analyzer import EmotionAnalyzer


@pytest.mark.asyncio
async def test_surface_advice_but_vulnerable_switches_to_support():
    analyzer = EmotionAnalyzer(use_llm_assist=False)
    event = await analyzer.analyze(
        raw_text="也没什么，就是最近有点撑不住了。你先告诉我该怎么办吧。",
        user_id="u1",
        session_id="s1",
    )
    assert event.intent == "support"
    assert event.support_need >= 0.45
    assert event.vulnerability_signal >= 0.35


@pytest.mark.asyncio
async def test_restrained_vulnerability_gets_detected():
    analyzer = EmotionAnalyzer(use_llm_assist=False)
    event = await analyzer.analyze(
        raw_text="我没事，不用担心，就是这几天有点难过。",
        user_id="u2",
        session_id="s2",
    )
    assert event.intent == "support"
    assert event.vulnerability_signal > 0
    assert event.support_need > 0


@pytest.mark.asyncio
async def test_llm_assist_can_raise_empathy_need():
    analyzer = EmotionAnalyzer(use_llm_assist=True)
    with patch("aimemo.emotion.analyzer.analyze_emotion_signals", new_callable=AsyncMock) as mocked:
        mocked.return_value = {
            "intent": "support",
            "support_need": 0.4,
            "empathy_need": 0.95,
            "vulnerability_signal": 0.8,
            "attachment_signal": 0.0,
            "conflict_signal": 0.0,
        }
        event = await analyzer.analyze(
            raw_text="给我一个计划吧。",
            user_id="u3",
            session_id="s3",
        )

    assert event.intent == "support"
    assert event.support_need > 0.4
    assert event.vulnerability_signal > 0.3
