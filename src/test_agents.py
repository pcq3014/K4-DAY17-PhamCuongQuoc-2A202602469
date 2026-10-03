from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import recall_points
from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore, extract_profile_updates

ROOT = Path(__file__).resolve().parent.parent


def make_config(tmp_path: Path):
    """Isolated config: state lives in tmp_path and compaction kicks in quickly."""

    config = load_config(ROOT)
    return dataclasses.replace(
        config,
        state_dir=tmp_path / "state",
        compact_threshold_tokens=300,
        compact_keep_messages=4,
        offline=True,
    )


def stress_turns() -> list[str]:
    return json.loads((ROOT / "data" / "advanced_long_context.json").read_text(encoding="utf-8"))[0]["turns"]


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.file_size("dungct") == 0
    assert "User.md" in store.read_text("dungct")  # empty default profile

    path = store.write_text("dungct", "# User.md\n- name: DũngCT\n- location: Đà Nẵng\n")
    assert path.exists() and path.name == "User.md"
    assert "Đà Nẵng" in store.read_text("dungct")

    assert store.edit_text("dungct", "Đà Nẵng", "Huế") is True
    assert "Huế" in store.read_text("dungct") and "Đà Nẵng" not in store.read_text("dungct")
    assert store.edit_text("dungct", "không tồn tại", "x") is False
    assert store.file_size("dungct") == len(store.read_text("dungct").encode("utf-8"))

    # User ids are sanitised before becoming paths.
    assert store.path_for("../evil user").parent.parent == tmp_path / "profiles"


def test_user_markdown_conflict_keeps_only_newest_fact(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.upsert_fact("dungct", "location", "Đà Nẵng")
    assert store.upsert_fact("dungct", "location", "Huế")

    assert store.facts("dungct")["location"] == "Huế"
    assert "Đà Nẵng" not in store.prompt_view("dungct")  # old value never reaches the prompt
    assert "Đà Nẵng → Huế" in store.read_text("dungct")  # ...but stays in the audit log


def test_extraction_handles_corrections_noise_and_questions() -> None:
    assert extract_profile_updates("Chào bạn, mình tên là DũngCT.") == {"name": "DũngCT"}
    correction = "À, mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa."
    assert extract_profile_updates(correction)["location"] == "Huế"
    switch = "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer."
    assert extract_profile_updates(switch)["profession"] == "MLOps engineer"

    # Noise must not become a fact (confidence threshold).
    joke = "Có lúc mình đùa rằng hay là chuyển sang product manager cho đỡ phải canh pipeline."
    assert "profession" not in extract_profile_updates(joke)
    assert "location" not in extract_profile_updates("Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày.")
    # Questions are not facts.
    assert extract_profile_updates("Hiện tại mình làm nghề gì và mình còn ở Huế không?") == {}


def test_compact_trigger(tmp_path: Path) -> None:
    manager = CompactMemoryManager(threshold_tokens=50, keep_messages=2)
    for index in range(6):
        manager.append("t1", "user", f"Tin số {index}: " + "nội dung khá dài " * 10)
    state = manager.context("t1")
    assert manager.compaction_count("t1") >= 1
    assert len(state["messages"]) <= 2
    assert "Tin số 0" in state["summary"]

    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for turn in stress_turns():
        agent.reply("dungct_stress", "stress", turn)
    assert agent.compaction_count("stress") >= 2
    assert len(agent.compact_memory.context("stress")["messages"]) <= 4

    # Short chats stay below the threshold: no compaction.
    agent.reply("dungct_stress", "short", "Chào bạn, hôm nay mình hơi bận.")
    assert agent.compaction_count("short") == 0


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    turns = [
        "Chào bạn, mình tên là DũngCT.",
        "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.",
        "Đồ uống yêu thích là cà phê sữa đá.",
        "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.",
    ]
    for turn in turns:
        baseline.reply("dungct", "session-1", turn)
        advanced.reply("dungct", "session-1", turn)

    # Within the same thread, baseline does remember.
    assert "DũngCT" in baseline.reply("dungct", "session-1", "Bạn có thể nhắc lại tên mình không?")["response"]

    question = "Mình tên gì, làm nghề gì và đồ uống yêu thích là gì?"
    expected = ["DũngCT", "MLOps engineer", "cà phê sữa đá"]
    advanced_answer = advanced.reply("dungct", "session-2", question)["response"]
    baseline_answer = baseline.reply("dungct", "session-2", question)["response"]

    assert recall_points(advanced_answer, expected) == 1.0
    assert "backend engineer" not in advanced_answer  # correction applied, stale fact not returned
    assert recall_points(baseline_answer, expected) == 0.0

    # Persistence survives a brand new agent instance (e.g. process restart).
    fresh = AdvancedAgent(config, force_offline=True)
    assert "DũngCT" in fresh.reply("dungct", "session-3", "Nhắc lại giúp mình tên của mình.")["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    turns = stress_turns()
    for turn in turns:
        baseline.reply("dungct_stress", "long", turn)
        advanced.reply("dungct_stress", "long", turn)

    assert advanced.compaction_count("long") > 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long") * 0.75

    # The per-turn prompt of baseline keeps growing; advanced's stays bounded.
    before = baseline.prompt_token_usage("long")
    baseline.reply("dungct_stress", "long", "Ok.")
    baseline_last_turn = baseline.prompt_token_usage("long") - before
    before = advanced.prompt_token_usage("long")
    advanced.reply("dungct_stress", "long", "Ok.")
    advanced_last_turn = advanced.prompt_token_usage("long") - before
    assert advanced_last_turn < baseline_last_turn / 2
