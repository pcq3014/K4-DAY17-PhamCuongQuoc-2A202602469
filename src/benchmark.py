from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config

COLUMNS = [
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
]


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int
    details: list[dict[str, Any]] = field(default_factory=list, repr=False)


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of conversations")
    return data


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def _coverage(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    hits = sum(1 for item in expected if _norm(item) in _norm(answer))
    return hits / len(expected)


def recall_points(answer: str, expected: list[str]) -> float:
    """1 if every expected fact appears, 0.5 if some do, 0 if none."""

    coverage = _coverage(answer, expected)
    if coverage == 1.0:
        return 1.0
    return 0.5 if coverage > 0 else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight 0-5 quality score for offline mode.

    - 3.0 for factual coverage of the expected facts
    - 1.0 for being concise (<= 300 chars, linearly penalised up to 900)
    - 0.5 for structure (bullet lines, matching the user's preferred style)
    - 0.5 for not admitting missing memory
    """

    coverage = _coverage(answer, expected)
    length = len(answer.strip())
    concise = 1.0 if length <= 300 else max(0.0, 1 - (length - 300) / 600)
    structured = 0.5 if re.search(r"^\s*[-*•]\s", answer, re.MULTILINE) else 0.0
    honest_gap = 0.0 if "chưa có" in _norm(answer) else 0.5
    return round(3 * coverage + concise + structured + honest_gap, 2)


def llm_judge_quality(judge, question: str, answer: str, expected: list[str]) -> float | None:
    """Ask the judge model for a 1-5 score; None if the call/parse fails."""

    prompt = (
        "Chấm điểm câu trả lời của trợ lý từ 1 đến 5 (5 = đúng đủ fact, ngắn gọn, đúng style). "
        f"Câu hỏi: {question}\nFact kỳ vọng: {', '.join(expected)}\nCâu trả lời: {answer}\n"
        "Chỉ trả về một con số."
    )
    try:
        reply = judge.invoke(prompt)
        match = re.search(r"[1-5](?:\.\d+)?", str(getattr(reply, "content", reply)))
        return float(match.group(0)) if match else None
    except Exception:  # noqa: BLE001 - judge is best-effort
        return None


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config, judge=None) -> BenchmarkRow:
    """Feed every turn, then ask recall questions in a fresh thread; aggregate the metrics."""

    users = sorted({conv["user_id"] for conv in conversations})
    initial_sizes = {user: agent.memory_file_size(user) for user in users}
    threads: list[str] = []
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    details: list[dict[str, Any]] = []

    for conv in conversations:
        main_thread = f"{conv['id']}::main"
        threads.append(main_thread)
        for turn in conv["turns"]:
            agent.reply(conv["user_id"], main_thread, turn)

        for index, item in enumerate(conv.get("recall_questions", []), start=1):
            recall_thread = f"{conv['id']}::recall-{index}"  # fresh thread = new session
            threads.append(recall_thread)
            answer = agent.reply(conv["user_id"], recall_thread, item["question"])["response"]
            expected = item["expected_contains"]
            score = recall_points(answer, expected)
            quality = llm_judge_quality(judge, item["question"], answer, expected) if judge else None
            quality = quality if quality is not None else heuristic_quality(answer, expected)
            recall_scores.append(score)
            quality_scores.append(quality)
            details.append(
                {"conversation": conv["id"], "question": item["question"], "answer": answer, "recall": score, "quality": quality}
            )

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(t) for t in threads),
        prompt_tokens_processed=sum(agent.prompt_token_usage(t) for t in threads),
        recall_score=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        response_quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        memory_growth_bytes=sum(agent.memory_file_size(u) - initial_sizes[u] for u in users),
        compactions=sum(agent.compaction_count(t) for t in threads),
        details=details,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    table = [
        [
            row.agent_name,
            f"{row.agent_tokens_only:,}",
            f"{row.prompt_tokens_processed:,}",
            f"{row.recall_score:.0%}",
            f"{row.response_quality:.2f}/5",
            f"{row.memory_growth_bytes:,}",
            str(row.compactions),
        ]
        for row in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=COLUMNS, tablefmt="github")
    except ImportError:
        lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
        lines += ["| " + " | ".join(cells) + " |" for cells in table]
        return "\n".join(lines)


def _ratio_note(baseline: BenchmarkRow, advanced: BenchmarkRow) -> str:
    def pct(new: int, old: int) -> str:
        if old == 0:
            return "n/a"
        delta = (new - old) / old
        return f"{delta:+.0%}"

    return (
        f"Advanced vs Baseline → agent tokens {pct(advanced.agent_tokens_only, baseline.agent_tokens_only)}, "
        f"prompt tokens {pct(advanced.prompt_tokens_processed, baseline.prompt_tokens_processed)}, "
        f"recall {baseline.recall_score:.0%} → {advanced.recall_score:.0%}"
    )


def run_suite(title: str, dataset: Path, config, live: bool, verbose: bool, judge=None) -> list[BenchmarkRow]:
    conversations = load_conversations(dataset)
    suite_state = config.state_dir / "benchmark" / dataset.stem
    shutil.rmtree(suite_state, ignore_errors=True)  # every run starts from an empty User.md
    suite_config = dataclasses.replace(config, state_dir=suite_state)

    rows = [
        run_agent_benchmark("Baseline", BaselineAgent(suite_config, force_offline=not live), conversations, config, judge),
        run_agent_benchmark("Advanced", AdvancedAgent(suite_config, force_offline=not live), conversations, config, judge),
    ]
    turns = sum(len(c["turns"]) for c in conversations)
    questions = sum(len(c.get("recall_questions", [])) for c in conversations)
    print(f"\n## {title}")
    print(f"_{dataset.name}: {len(conversations)} conversation(s), {turns} turns, {questions} recall questions_\n")
    print(format_rows(rows))
    print("\n" + _ratio_note(rows[0], rows[1]))
    if verbose:
        for row in rows:
            print(f"\n### {row.agent_name} answers")
            for item in row.details:
                answer = item["answer"].replace("\n", " | ")
                print(f"- [{item['conversation']}] recall={item['recall']} q={item['quality']}: {answer}")
    return rows


def main() -> None:
    """Run the Standard benchmark and the Long-Context Stress benchmark for both agents."""

    parser = argparse.ArgumentParser(description="Day 17 memory benchmark: Baseline vs Advanced")
    parser.add_argument("--live", action="store_true", help="use the configured LLM provider instead of offline mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="print every recall answer")
    args = parser.parse_args()

    config = load_config(Path(__file__).resolve().parent.parent)
    live = args.live and config.model.is_configured
    if args.live and not live:
        print("! --live requested but no provider credentials found; falling back to offline mode.")

    judge = None
    if live and config.judge_model.is_configured:
        from model_provider import build_chat_model

        judge = build_chat_model(config.judge_model)

    print("# Day 17 Memory Benchmark")
    print(
        f"mode={'live ' + config.model.provider + '/' + config.model.model_name if live else 'offline (deterministic)'} · "
        f"compact_threshold={config.compact_threshold_tokens} tokens · keep={config.compact_keep_messages} messages · "
        f"confidence_threshold={config.memory_confidence_threshold}"
    )
    run_suite("Standard Benchmark", config.data_dir / "conversations.json", config, live, args.verbose, judge)
    run_suite("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json", config, live, args.verbose, judge)


if __name__ == "__main__":
    main()
