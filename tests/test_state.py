"""The state object, its reducers, and the Settings isolation property."""
import pytest

from cricketmind import DEFAULT_SETTINGS, Settings, new_state, upsert_findings


def test_upsert_keeps_one_slot_per_agent():
    """Different agents pile up; the same agent replaces itself."""
    a1 = {"agent": "n11b", "text": "first"}
    a2 = {"agent": "n11b", "text": "corrected"}
    b = {"agent": "n11c", "text": "other"}

    assert upsert_findings([a1], [b]) == [a1, b]          # different agents accumulate
    assert upsert_findings([a1], [a2]) == [a2]            # same agent replaces
    assert upsert_findings([a1, b], [a2]) == [a2, b]      # and keeps its position


def test_upsert_tolerates_none():
    """LangGraph passes None on the very first write."""
    assert upsert_findings(None, None) == []
    assert upsert_findings(None, [{"agent": "x"}]) == [{"agent": "x"}]


def test_new_state_starts_collections_empty():
    """A reducer must never receive None on the first write."""
    s = new_state("clip.mp4", "q")
    for key in ("stats_hits", "context_hits", "active_agents",
                "agent_findings", "faith_failures"):
        assert s[key] == [], key
    assert s["retry_count"] == 0 and s["regen_count"] == 0
    assert s["status"] == "ok"


def test_new_state_accepts_no_clip():
    assert new_state(None, "what if Rohit opened?")["video_path"] is None


def test_settings_default_when_absent():
    assert new_state("c", "q")["settings"] is DEFAULT_SETTINGS


def test_settings_normalises_sequences():
    """Lists and sets may be passed; they are stored as tuples and frozensets."""
    s = Settings(confidence=[0.4, 0.9], lie_once={"a"}, lie_always=["b"])
    assert s.confidence == (0.4, 0.9)
    assert s.lie_once == frozenset({"a"}) and s.lie_always == frozenset({"b"})


def test_settings_are_frozen():
    with pytest.raises(Exception):
        Settings().confidence = 0.1


def test_settings_reject_unknown_router():
    with pytest.raises(AssertionError):
        Settings(router_mode="magic")


def test_with_returns_a_copy():
    base = Settings()
    changed = base.with_(trace=True)
    assert changed.trace and not base.trace
    assert changed.confidence == base.confidence
