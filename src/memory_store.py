from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 characters per token).

    Not tokenizer-exact, but stable across runs, which is what the offline
    benchmark needs to compare agents fairly.
    """

    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, math.ceil(len(stripped) / 4))


# ---------------------------------------------------------------------------
# Structured profile (entity extraction target)
# ---------------------------------------------------------------------------

FIELD_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "drink": "Đồ uống yêu thích",
    "food": "Món ăn yêu thích",
    "pet": "Thú cưng",
    "response_style": "Style trả lời",
    "interests": "Mối quan tâm kỹ thuật",
    "hobbies": "Sở thích ngoài giờ",
}
SINGLE_VALUE_FIELDS = ("name", "location", "profession", "drink", "food", "pet", "response_style")
MULTI_VALUE_FIELDS = ("interests", "hobbies")

# Recency decay applied to multi-valued entries: score = mentions * DECAY ** (revisions since last mention).
INTEREST_DECAY = 0.9
TOP_INTERESTS = 4
MAX_CORRECTIONS_LOGGED = 5


@dataclass
class FactRecord:
    value: str
    confidence: float
    mentions: int = 1
    revision: int = 0


@dataclass
class TopicRecord:
    mentions: int = 1
    revision: int = 0


@dataclass
class UserProfile:
    user_id: str
    revision: int = 0
    facts: dict[str, FactRecord] = field(default_factory=dict)
    topics: dict[str, dict[str, TopicRecord]] = field(default_factory=dict)
    corrections: list[str] = field(default_factory=list)

    def ranked(self, key: str, top_k: int = TOP_INTERESTS) -> list[str]:
        """Rank multi-valued entries by mention frequency with recency decay."""

        entries = self.topics.get(key, {})
        scored = sorted(
            entries.items(),
            key=lambda item: (
                -(item[1].mentions * INTEREST_DECAY ** max(0, self.revision - item[1].revision)),
                -item[1].revision,
            ),
        )
        return [name for name, _ in scored[:top_k]]

    def as_dict(self) -> dict[str, str]:
        data = {key: record.value for key, record in self.facts.items()}
        for key in MULTI_VALUE_FIELDS:
            ranked = self.ranked(key)
            if ranked:
                data[key] = ", ".join(ranked)
        return data


_FACT_LINE = re.compile(r"^- (\w+): (.*?) _\(conf ([\d.]+) · seen (\d+) · rev (\d+)\)_$")
_TOPIC_LINE = re.compile(r"^- (.*?) _\(seen (\d+) · rev (\d+)\)_$")
_REVISION_LINE = re.compile(r"<!-- revision: (\d+) -->")


def render_profile(profile: UserProfile) -> str:
    lines = [f"# User.md — {profile.user_id}", "", f"<!-- revision: {profile.revision} -->", "", "## Profile"]
    for key in SINGLE_VALUE_FIELDS:
        record = profile.facts.get(key)
        if record:
            lines.append(
                f"- {key}: {record.value} _(conf {record.confidence:.2f} · seen {record.mentions} · rev {record.revision})_"
            )
    for key in MULTI_VALUE_FIELDS:
        entries = profile.topics.get(key)
        if not entries:
            continue
        lines += ["", f"## {key.capitalize()}"]
        # Store everything, ordered by decayed score; the prompt only uses the top entries.
        for name in profile.ranked(key, top_k=len(entries)):
            record = entries[name]
            lines.append(f"- {name} _(seen {record.mentions} · rev {record.revision})_")
    if profile.corrections:
        lines += ["", "## Corrections (đã thay thế, không dùng làm fact hiện tại)"]
        lines += [f"- {entry}" for entry in profile.corrections]
    return "\n".join(lines) + "\n"


def parse_profile(user_id: str, text: str) -> UserProfile:
    profile = UserProfile(user_id=user_id)
    match = _REVISION_LINE.search(text)
    if match:
        profile.revision = int(match.group(1))
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            section = line[3:].split(" ")[0].lower()
            continue
        if not line.startswith("- "):
            continue
        if section == "profile":
            m = _FACT_LINE.match(line)
            if m:
                key, value, conf, seen, rev = m.groups()
                profile.facts[key] = FactRecord(value, float(conf), int(seen), int(rev))
        elif section in MULTI_VALUE_FIELDS:
            m = _TOPIC_LINE.match(line)
            if m:
                name, seen, rev = m.groups()
                profile.topics.setdefault(section, {})[name] = TopicRecord(int(seen), int(rev))
        elif section == "corrections":
            profile.corrections.append(line[2:])
    return profile


# ---------------------------------------------------------------------------
# Persistent memory: User.md
# ---------------------------------------------------------------------------


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`: one markdown file per user id."""

    root_dir: Path
    confidence_threshold: float = 0.6

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id.strip()).strip("._") or "anonymous"
        return Path(self.root_dir) / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return render_profile(UserProfile(user_id=user_id))

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        path = self.path_for(user_id)
        if not path.exists() or not search_text:
            return False
        content = path.read_text(encoding="utf-8")
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    # -- structured helpers -------------------------------------------------

    def load(self, user_id: str) -> UserProfile:
        return parse_profile(user_id, self.read_text(user_id))

    def save(self, profile: UserProfile) -> Path:
        return self.write_text(profile.user_id, render_profile(profile))

    def facts(self, user_id: str) -> dict[str, str]:
        return self.load(user_id).as_dict()

    def prompt_view(self, user_id: str) -> str:
        """What actually goes into the prompt: current facts only, no audit metadata or corrections log."""

        facts = self.facts(user_id)
        if not facts:
            return ""
        lines = [f"- {FIELD_LABELS[key]}: {value}" for key, value in facts.items() if key in FIELD_LABELS]
        return "User profile (User.md):\n" + "\n".join(lines)

    def upsert_fact(self, user_id: str, key: str, value: str, confidence: float = 0.9) -> bool:
        return bool(self.apply_candidates(user_id, [FactCandidate(key, value, confidence)]))

    def apply_candidates(self, user_id: str, candidates: list["FactCandidate"]) -> list[str]:
        """Apply extracted candidates with confidence gating + conflict handling.

        Returns human-readable change notes (empty when User.md was not touched).
        """

        accepted = [c for c in candidates if c.confidence >= self.confidence_threshold]
        if not accepted:
            return []
        profile = self.load(user_id)
        profile.revision += 1
        changes: list[str] = []
        for cand in accepted:
            if cand.key in MULTI_VALUE_FIELDS:
                entries = profile.topics.setdefault(cand.key, {})
                existing = next((k for k in entries if k.lower() == cand.value.lower()), None)
                if existing:
                    entries[existing].mentions += 1
                    entries[existing].revision = profile.revision
                else:
                    entries[cand.value] = TopicRecord(1, profile.revision)
                    changes.append(f"+{cand.key}: {cand.value}")
                continue

            current = profile.facts.get(cand.key)
            value = merge_style(current.value, cand.value) if cand.key == "response_style" and current else cand.value
            if current and current.value.lower() == value.lower():
                current.mentions += 1
                current.confidence = max(current.confidence, cand.confidence)
                current.revision = profile.revision
                continue
            if current and cand.key != "response_style":
                # Conflict: the newest confident statement wins; old value goes to an audit log only.
                profile.corrections.append(f"{cand.key}: {current.value} → {value} (rev {profile.revision})")
                profile.corrections = profile.corrections[-MAX_CORRECTIONS_LOGGED:]
            mentions = current.mentions + 1 if current and cand.key == "response_style" else 1
            profile.facts[cand.key] = FactRecord(value, cand.confidence, mentions, profile.revision)
            changes.append(f"{cand.key}: {value}")
        self.save(profile)
        return changes


# ---------------------------------------------------------------------------
# Fact extraction (rule-based entity extraction with confidence)
# ---------------------------------------------------------------------------


@dataclass
class FactCandidate:
    key: str
    value: str
    confidence: float
    reason: str = ""


KNOWN_PLACES = sorted(
    [
        "Hà Nội", "Hồ Chí Minh", "TP.HCM", "Sài Gòn", "Đà Nẵng", "Huế", "Hải Phòng", "Cần Thơ",
        "Nha Trang", "Đà Lạt", "Hội An", "Quảng Nam", "Quảng Ngãi", "Quy Nhơn", "Vũng Tàu",
        "Bắc Ninh", "Nghệ An", "Vinh", "Thanh Hóa", "Singapore", "Tokyo", "Seoul", "Bangkok",
    ],
    key=len,
    reverse=True,
)
_PLACE_ALT = "|".join(re.escape(p) for p in KNOWN_PLACES)
_LOCATION_PATTERNS = [
    re.compile(rf"(?:\bở|\btại|\bsang)\s+({_PLACE_ALT})(?![\wÀ-ỹ])"),
    re.compile(rf"nơi ở(?: hiện tại)? (?:là|ở)\s+({_PLACE_ALT})(?![\wÀ-ỹ])", re.IGNORECASE),
]
_ROLE_PATTERN = re.compile(
    r"(?<![\w-])([A-Za-z][A-Za-z0-9\-]*\s+(?:engineer|developer|manager|scientist|analyst|designer|researcher))\b",
    re.IGNORECASE,
)
_NAME_PATTERNS = [
    re.compile(r"(?:mình|tôi|em)\s+tên\s+là\s+(.+)", re.IGNORECASE),
    re.compile(r"tên\s+(?:của\s+)?(?:mình|tôi|em)\s+là\s+(.+)", re.IGNORECASE),
]
_DRINK_PATTERNS = [
    (re.compile(r"đồ uống (?:yêu thích|ưa thích|ruột)(?: của mình)? là ([^.,;!?]+)", re.IGNORECASE), 0.9),
    (re.compile(r"(?:vẫn|hay|thường) uống ([^.,;!?]+)", re.IGNORECASE), 0.7),
]
_FOOD_PATTERNS = [
    (re.compile(r"món (?:ăn )?(?:yêu thích|ưa thích|ruột)(?: của mình)? là ([^.,;!?]+)", re.IGNORECASE), 0.9),
]
_PET_PATTERN = re.compile(r"\bnuôi (?:một |1 )?(?:bé |con |chú |em )?([^\s.,;!?]+)(?: tên ([^\s.,;!?]+))?", re.IGNORECASE)
_PHRASE_STOPS = (" như cũ", " nhưng", " mỗi", " vào", " để", " và thấy", " khi")

INTEREST_TOPICS = sorted(
    [
        "async Python", "Python", "AI ứng dụng", "AI agent", "MLOps", "RAG", "evaluation",
        "memory architecture", "memory compaction", "benchmark memory", "LangChain", "LangGraph",
    ],
    key=len,
    reverse=True,
)
HOBBY_TOPICS = ["chạy bộ", "lo-fi", "chụp ảnh", "đi bộ"]
_INTEREST_VERBS = ("thích", "quan tâm", "đang học", "học thêm", "ôn lại", "đang đọc", "đam mê")
_STYLE_TRIGGERS = ("trả lời", "giải thích", "style")
_STYLE_VERBS = ("muốn", "thích", "hãy", "giữ", "ưu tiên", "nên")

# Cues that make a whole sentence unreliable as a profile fact.
_NOISE_CUES = (
    "đùa", "chỉ là nơi", "đi họp", "bay ra họp", "công tác", "du lịch", "ví dụ cũ",
    "đừng lấy", "đừng nói", "thông tin cũ", "giả sử", "hay là",
)
# Cues that lower confidence (conditional / tentative statements).
_HEDGE_CUES = ("nếu", "có lẽ", "chắc là", "định ", "cân nhắc", "dự định")
# Cues that a statement is a fresh correction: newest wins, slightly higher confidence.
_CORRECTION_CUES = ("đính chính", "chuyển sang", "cập nhật", "thực ra", "giờ ", "hiện tại", "từ tuần này")
# Cues right before a mention that negate it or mark it as outdated.
_NEGATION_CUES = ("không còn", "không phải", "chứ không", "không ở", "đừng", "lúc đầu", "trước đó", "trước đây")
# "Nhắc lại lần cuối cho chắc: ..." is the user restating facts, not asking for them.
_QUESTION_START = re.compile(r"^(?:nhắc lại(?! lần cuối)|tóm tắt|bạn (?:có )?biết|cho mình hỏi|bạn thử nhớ lại)", re.IGNORECASE)
_REQUEST_PATTERN = re.compile(r"nhắc lại giúp|giúp mình:", re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def is_question(text: str) -> bool:
    """A recall question / request (vs. a statement that carries facts)."""

    if _REQUEST_PATTERN.search(text):
        return True
    return any(s.endswith("?") or _QUESTION_START.match(s) for s in split_sentences(text))


def _sentence_confidence(sentence: str, base: float, allow_hedge: bool = False) -> float:
    lowered = sentence.lower()
    if any(cue in lowered for cue in _NOISE_CUES):
        return 0.2
    conf = base
    if not allow_hedge and any(cue in lowered for cue in _HEDGE_CUES):
        conf -= 0.3
    if any(cue in lowered for cue in _CORRECTION_CUES):
        conf += 0.05
    return round(min(conf, 0.99), 2)


def _is_negated(sentence: str, start: int, window: int = 22) -> bool:
    before = sentence[max(0, start - window):start].lower()
    return any(cue in before for cue in _NEGATION_CUES)


def _trim_phrase(value: str) -> str:
    value = value.strip()
    lowered = value.lower()
    cut = len(value)
    for stop in _PHRASE_STOPS:
        idx = lowered.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return value[:cut].strip()


def _clean_name(raw: str) -> str:
    tokens = []
    for token in raw.split():
        word = token.strip(".,;:!?\"'()")
        if not word or not word[0].isupper():
            break
        tokens.append(word)
        if token[-1:] in ".,;:!?":
            break
    return " ".join(tokens[:4])


STYLE_FEATURES = [
    ("brevity", re.compile(r"ngắn gọn|\bgọn\b|bullet ngắn|\bngắn\b", re.IGNORECASE), lambda m: "ngắn gọn"),
    ("clarity", re.compile(r"rõ ý", re.IGNORECASE), lambda m: "rõ ý"),
    ("bullets", re.compile(r"(\d+)\s*bullet", re.IGNORECASE), lambda m: f"{m.group(1)} bullet"),
    ("bullets", re.compile(r"\bbullet\b", re.IGNORECASE), lambda m: "dạng bullet"),
    ("examples", re.compile(r"ví dụ (thực tế|thực chiến)", re.IGNORECASE), lambda m: f"có ví dụ {m.group(1).lower()}"),
    ("tradeoff", re.compile(r"trade-?off", re.IGNORECASE), lambda m: "so sánh trade-off"),
]
_STYLE_ORDER = ["brevity", "clarity", "bullets", "examples", "tradeoff"]


def _style_slots(text: str) -> dict[str, str]:
    slots: dict[str, str] = {}
    for slot, pattern, render in STYLE_FEATURES:
        if slot in slots:
            continue  # first (more specific) pattern for a slot wins, e.g. "3 bullet" over "dạng bullet"
        match = pattern.search(text)
        if match:
            slots[slot] = render(match)
    return slots


def _render_style(slots: dict[str, str]) -> str:
    return ", ".join(slots[s] for s in _STYLE_ORDER if s in slots)


def merge_style(old: str, new: str) -> str:
    """Merge style preferences slot by slot; newer statements override the same slot."""

    slots = _style_slots(old)
    slots.update(_style_slots(new))
    return _render_style(slots)


def _match_topics(sentence: str, topics: list[str]) -> list[str]:
    found: list[str] = []
    masked = sentence
    for topic in topics:
        pattern = re.compile(rf"(?<![\wÀ-ỹ]){re.escape(topic)}(?![\wÀ-ỹ])", re.IGNORECASE)
        if pattern.search(masked):
            found.append(topic)
            masked = pattern.sub(" ", masked)
    return found


def extract_fact_candidates(message: str) -> list[FactCandidate]:
    """Extract profile facts with a confidence score from one user message."""

    if not message or is_question(message):
        return []

    candidates: list[FactCandidate] = []
    for sentence in split_sentences(message):
        if sentence.endswith("?"):
            continue
        lowered = sentence.lower()

        for pattern in _NAME_PATTERNS:
            m = pattern.search(sentence)
            if m:
                name = _clean_name(m.group(1))
                if name:
                    candidates.append(FactCandidate("name", name, _sentence_confidence(sentence, 0.95), "name pattern"))
                break

        locations = []
        for pattern in _LOCATION_PATTERNS:
            for m in pattern.finditer(sentence):
                if not _is_negated(sentence, m.start()):
                    locations.append((m.start(), m.group(1)))
        if locations:
            place = max(locations)[1]  # the last non-negated mention in the sentence wins
            candidates.append(FactCandidate("location", place, _sentence_confidence(sentence, 0.85), "location"))

        roles = []
        for m in _ROLE_PATTERN.finditer(sentence):
            before = sentence[max(0, m.start() - 22):m.start()].lower()
            if _is_negated(sentence, m.start()):
                continue
            if re.search(r"\b(làm|là|sang|nghề)\b", before):
                roles.append((m.start(), m.group(1)))
        if roles:
            role = max(roles)[1]
            candidates.append(FactCandidate("profession", role, _sentence_confidence(sentence, 0.85), "profession"))

        for pattern, base in _DRINK_PATTERNS:
            m = pattern.search(sentence)
            if m and _trim_phrase(m.group(1)):
                candidates.append(FactCandidate("drink", _trim_phrase(m.group(1)), _sentence_confidence(sentence, base)))
                break

        for pattern, base in _FOOD_PATTERNS:
            m = pattern.search(sentence)
            if m and _trim_phrase(m.group(1)):
                candidates.append(FactCandidate("food", _trim_phrase(m.group(1)), _sentence_confidence(sentence, base)))
                break

        m = _PET_PATTERN.search(sentence)
        if m:
            pet = m.group(1) + (f" tên {m.group(2)}" if m.group(2) else "")
            candidates.append(FactCandidate("pet", pet, _sentence_confidence(sentence, 0.85), "pet"))

        if any(t in lowered for t in _STYLE_TRIGGERS) and any(v in lowered for v in _STYLE_VERBS):
            style = _render_style(_style_slots(sentence))
            if style:
                # Conditional phrasing ("nếu bạn giải thích, hãy ...") is still a real preference.
                candidates.append(
                    FactCandidate("response_style", style, _sentence_confidence(sentence, 0.8, allow_hedge=True))
                )

        if any(v in lowered for v in _INTEREST_VERBS):
            for topic in _match_topics(sentence, INTEREST_TOPICS):
                candidates.append(FactCandidate("interests", topic, _sentence_confidence(sentence, 0.75)))
            for hobby in _match_topics(sentence, HOBBY_TOPICS):
                candidates.append(FactCandidate("hobbies", hobby, _sentence_confidence(sentence, 0.7)))

    # Within one message, keep the last candidate per single-valued key.
    deduped: dict[tuple[str, str], FactCandidate] = {}
    for cand in candidates:
        slot = (cand.key, cand.value if cand.key in MULTI_VALUE_FIELDS else "")
        deduped.pop(slot, None)
        deduped[slot] = cand
    return list(deduped.values())


def extract_profile_updates(message: str, threshold: float = 0.6) -> dict[str, str]:
    """Convert raw user text into stable profile facts (only confident ones)."""

    updates: dict[str, str] = {}
    for cand in extract_fact_candidates(message):
        if cand.confidence < threshold:
            continue
        if cand.key in MULTI_VALUE_FIELDS:
            updates[cand.key] = f"{updates[cand.key]}, {cand.value}" if cand.key in updates else cand.value
        else:
            updates[cand.key] = cand.value
    return updates


# ---------------------------------------------------------------------------
# Answering from memory (shared by both agents in offline mode)
# ---------------------------------------------------------------------------

_FIELD_CUES = [
    ("name", ("tên",)),
    ("profession", ("nghề",)),
    ("location", ("ở đâu", "nơi ở", "còn ở", "đang ở")),
    ("drink", ("đồ uống", "uống gì")),
    ("food", ("món ăn", "ăn gì")),
    ("pet", ("nuôi",)),
    ("response_style", ("style", "kiểu trả lời", "trả lời như thế nào", "cách trả lời")),
    ("interests", ("quan tâm", "thích gì")),
]
_PROFILE_SUMMARY_CUES = ("là ai", "tóm tắt", "mô tả", "về mình")


def requested_fields(question: str) -> list[str]:
    lowered = question.lower()
    fields = [key for key, cues in _FIELD_CUES if any(cue in lowered for cue in cues)]
    if any(cue in lowered for cue in _PROFILE_SUMMARY_CUES):
        for key in ("name", "profession", "interests"):
            if key not in fields:
                fields.append(key)
    return fields


def compose_answer(question: str, facts: dict[str, str]) -> str | None:
    """Answer a recall question from a facts dict; None when nothing is being asked."""

    fields = requested_fields(question)
    if not fields:
        return None
    lines = [f"- {FIELD_LABELS[key]}: {facts[key]}" for key in fields if facts.get(key)]
    missing = [FIELD_LABELS[key].lower() for key in fields if not facts.get(key)]
    if missing:
        lines.append(f"- Chưa có trong bộ nhớ: {', '.join(missing)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compact memory
# ---------------------------------------------------------------------------


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary: one short line per user message (first sentence, truncated)."""

    lines = []
    for message in messages:
        if message.get("role") != "user":
            continue
        sentences = split_sentences(message.get("content", ""))
        if not sentences:
            continue
        first = sentences[0]
        if len(first) > 110:
            first = first[:107].rstrip() + "..."
        lines.append(f"- user: {first}")
    return "\n".join(lines[-max_items:])


@dataclass
class CompactMemoryManager:
    """Short-term memory per thread with automatic compaction.

    - Keeps the most recent `keep_messages` in full.
    - When summary + messages exceed `threshold_tokens`, older messages are folded
      into a bounded summary (at most `summary_max_items` lines).
    - Tracks compactions and generated summary tokens for benchmarking.
    """

    threshold_tokens: int
    keep_messages: int
    summary_max_items: int = 6
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
                "summarized_messages": 0,
                "summary_tokens_generated": 0,
            }
        return self.state[thread_id]

    def thread_tokens(self, thread_id: str) -> int:
        thread = self._thread(thread_id)
        total = estimate_tokens(thread["summary"])
        return total + sum(estimate_tokens(m["content"]) for m in thread["messages"])

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._thread(thread_id)
        thread["messages"].append({"role": role, "content": content})
        if self.thread_tokens(thread_id) > self.threshold_tokens and len(thread["messages"]) > self.keep_messages:
            self._compact(thread)

    def _compact(self, thread: dict[str, object]) -> None:
        messages = thread["messages"]
        cut = len(messages) - self.keep_messages
        old, recent = messages[:cut], messages[cut:]
        new_lines = summarize_messages(old, max_items=self.summary_max_items)
        merged = [line for line in (thread["summary"] + "\n" + new_lines).splitlines() if line.strip()]
        thread["summary"] = "\n".join(merged[-self.summary_max_items:])
        thread["messages"] = recent
        thread["compactions"] += 1
        thread["summarized_messages"] += len(old)
        thread["summary_tokens_generated"] += estimate_tokens(new_lines)

    def context(self, thread_id: str) -> dict[str, object]:
        return self._thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))
