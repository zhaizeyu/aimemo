"""Policy layer controlling explainable state transitions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aimemo.emotion.models import (
    EmotionEvent,
    EmotionState,
    RelationshipState,
    StyleMode,
    SupportMode,
)

FAST_BASELINE = {
    "mood_valence": 0.0,
    "energy": 0.5,
    "arousal": 0.4,
    "stress": 0.2,
    "loneliness": 0.3,
    "comfort_drive": 0.5,
    "guardedness": 0.2,
}



def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))



def apply_emotion_event(state: EmotionState, event: EmotionEvent) -> EmotionState:
    """Update fast variables with high responsiveness."""
    state.mood_valence = _clamp(state.mood_valence + 0.45 * event.sentiment, -1.0, 1.0)
    state.energy = _clamp(state.energy + 0.18 * event.playfulness_signal - 0.12 * event.support_need)
    state.arousal = _clamp(state.arousal + 0.25 * event.urgency + 0.15 * event.conflict_signal)
    state.stress = _clamp(
        state.stress
        + 0.35 * event.urgency
        + 0.30 * event.conflict_signal
        + 0.20 * event.support_need
        + 0.15 * event.vulnerability_signal
    )
    state.loneliness = _clamp(state.loneliness + 0.10 * event.attachment_signal - 0.08 * event.praise_signal)
    state.comfort_drive = _clamp(
        state.comfort_drive + 0.30 * event.support_need + 0.20 * event.vulnerability_signal
    )
    state.guardedness = _clamp(state.guardedness + 0.28 * event.conflict_signal)

    if event.conflict_signal > 0.45:
        state.support_mode = SupportMode.BOUNDARY
        state.style_mode = StyleMode.DIRECT
    elif event.support_need > 0.4 or event.vulnerability_signal > 0.4:
        state.support_mode = SupportMode.COMFORTING
        state.style_mode = StyleMode.WARM
    elif event.intent == "advice":
        state.support_mode = SupportMode.ADVISING
        state.style_mode = StyleMode.CALM
    elif event.praise_signal > 0.4:
        state.support_mode = SupportMode.CELEBRATING
        state.style_mode = StyleMode.WARM
    else:
        state.support_mode = SupportMode.NEUTRAL
        state.style_mode = StyleMode.CALM

    state.updated_at = datetime.now(UTC)
    return state



def apply_relationship_event(relation: RelationshipState, event: EmotionEvent) -> RelationshipState:
    """Update slow variables with small per-turn deltas."""
    relation.interaction_count += 1
    relation.last_interaction_at = datetime.now(UTC)

    relation.familiarity = _clamp(relation.familiarity + 0.015)
    relation.trust = _clamp(
        relation.trust + 0.02 * event.praise_signal - 0.03 * event.conflict_signal + 0.01 * event.support_need
    )
    relation.affection = _clamp(relation.affection + 0.02 * event.praise_signal - 0.015 * event.conflict_signal)
    relation.dependence = _clamp(relation.dependence + 0.01 * event.attachment_signal)
    relation.safety = _clamp(relation.safety + 0.015 * event.praise_signal - 0.04 * event.conflict_signal)
    relation.emotional_closeness = _clamp(
        relation.emotional_closeness
        + 0.015 * event.vulnerability_signal
        + 0.012 * event.praise_signal
        - 0.02 * event.conflict_signal
    )
    return relation



def decay_to_baseline(state: EmotionState, *, now: datetime | None = None) -> EmotionState:
    """Time-based recovery for fast variables."""
    ref_now = now or datetime.now(UTC)
    elapsed_hours = max((ref_now - state.updated_at).total_seconds() / 3600.0, 0.0)
    if elapsed_hours <= 0:
        return state

    # About half-way back to baseline every ~4 hours.
    alpha = min(0.7, elapsed_hours / 8.0)

    state.mood_valence = _clamp(
        state.mood_valence + (FAST_BASELINE["mood_valence"] - state.mood_valence) * alpha,
        -1.0,
        1.0,
    )
    for field in ("energy", "arousal", "stress", "loneliness", "comfort_drive", "guardedness"):
        current = getattr(state, field)
        target = FAST_BASELINE[field]
        setattr(state, field, _clamp(current + (target - current) * alpha))

    state.updated_at = ref_now
    return state



def should_materialize_pattern(recent_events: list[EmotionEvent], relation: RelationshipState) -> bool:
    """Only stable patterns should be materialized to long-term memory."""
    if len(recent_events) < 6:
        return False

    support_bias = sum(1 for e in recent_events if e.support_need >= 0.5)
    if support_bias >= 4:
        return True

    return relation.familiarity >= 0.55 and relation.interaction_count >= 12



def materialization_cooldown(last_iso: str | None, *, now: datetime | None = None) -> bool:
    if not last_iso:
        return False
    ref_now = now or datetime.now(UTC)
    last = datetime.fromisoformat(last_iso)
    return ref_now - last < timedelta(hours=24)
