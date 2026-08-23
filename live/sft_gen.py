"""Pure library for SFT dataset construction: passages, prompts, validation.

No Modal, no I/O side effects, no print -- modal_sft.py owns those. Every
threshold lives in sft_config.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import unicodedata
from dataclasses import asdict, dataclass, field

import sft_config as sc

log = logging.getLogger(__name__)

# =============================================================================
# Passages
# =============================================================================
_SENT_END = re.compile(r"(?<=[.!?])\s+")
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


@dataclass
class Passage:
    id: str
    source: str
    source_file: str
    line_no: int
    text: str
    n_tokens: int


def sample_line_numbers(total_lines: int, quota: int, seed: int) -> list[int]:
    """Evenly strided offsets with per-stride jitter -- spreads picks over a shard."""
    if quota >= total_lines:
        return list(range(total_lines))
    rng = random.Random(seed)
    stride = total_lines / quota
    picks = {min(total_lines - 1, int(i * stride + rng.random() * stride))
             for i in range(quota)}
    return sorted(picks)


def truncate_to_tokens(text: str, tok, max_tokens: int) -> tuple[str, int]:
    """Cut to <= max_tokens on a sentence boundary where possible."""
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text, len(ids)
    cut = tok.decode(ids[:max_tokens])
    parts = _SENT_END.split(cut)
    if len(parts) > 1:
        trimmed = " ".join(parts[:-1]).strip()
        if len(trimmed) >= 0.5 * len(cut):
            cut = trimmed
    return cut.strip(), len(tok.encode(cut, add_special_tokens=False))


def build_passage(source: str, source_file: str, line_no: int, raw: str, tok) -> Passage | None:
    text = _WS.sub(" ", raw).strip()
    if len(text) < sc.PASSAGE_MIN_CHARS:
        return None
    text, n_tokens = truncate_to_tokens(text, tok, sc.PASSAGE_MAX_TOKENS)
    if n_tokens < sc.PASSAGE_MIN_TOKENS:
        return None
    pid = f"{source}-{source_file.removesuffix('.txt')}-{line_no:07d}"
    return Passage(pid, source, source_file, line_no, text, n_tokens)


# =============================================================================
# Type allocation
# =============================================================================
def allocate(total: int, shares: dict[str, float]) -> dict[str, int]:
    """Largest-remainder split so the parts sum to exactly `total`."""
    exact = {k: total * v for k, v in shares.items()}
    out = {k: int(v) for k, v in exact.items()}
    for k, _ in sorted(exact.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True):
        if sum(out.values()) >= total:
            break
        out[k] += 1
    return out


def type_sequence(quota: int, seed: int) -> list[str]:
    counts = allocate(quota, sc.TYPE_MIX)
    seq = [t for t, n in counts.items() for _ in range(n)]
    random.Random(seed).shuffle(seq)
    return seq


# =============================================================================
# Generation prompt
# =============================================================================
_STYLE_HINTS: tuple[str, ...] = (
    "a 'what' question about a specific fact",
    "a 'who' question about a party, person, or entity",
    "a 'when' question about a date or time period",
    "a 'how much' question about an amount, figure, or percentage",
    "a yes/no question that the passage settles",
    "an 'explain' question answered in a short paragraph",
    "a 'which' question that picks between things named in the passage",
    "a 'why' question about a stated reason",
)

_LENGTH_HINTS: tuple[str, ...] = (
    "Answer in ONE short sentence.",
    "Answer in one or two sentences.",
    f"Answer in a short paragraph of at most {sc.ANSWER_MAX_WORDS} words.",
)

_TYPE_RULES: dict[str, str] = {
    "lookup": (
        "Write a LOOKUP question: the answer is stated explicitly in the passage "
        "(a party name, date, dollar amount, holding, statute, or figure). "
        "The answer must quote or closely paraphrase the passage."
    ),
    "reasoning": (
        "Write a REASONING question: a short why/how question that needs one or two "
        "inference steps, but every fact used must still come from the passage. "
        "Do not require outside case citations, outside filings, or world knowledge."
    ),
    "unanswerable": (
        "Write an UNANSWERABLE question: a question that is clearly ON TOPIC for this "
        "passage but that the passage does NOT answer. It must be a plausible thing a "
        "reader would ask, not nonsense, and it must not be answerable by guessing. "
        f'The answer field MUST be exactly: "{sc.REFUSAL_TEXT}"'
    ),
}

_GEN_INSTRUCTIONS = """You write training data for a small legal/financial assistant.

Read the passage below. Write ONE question-answer pair.

{type_rule}

Hard rules:
- The answer must be supported by the passage alone. Never add a fact that is not in it.
- Never mention "the passage", "the context", "the document", or "the excerpt" in the QUESTION. Ask as if the reader has the text in front of them.
- The question must be a real question, not a restatement or summary of the passage.
- {style_hint}
- {length_hint}
- Do not number the pair or add preamble.

Return JSON only: {{"question": "...", "answer": "..."}}

PASSAGE:
\"\"\"
{passage}
\"\"\"
"""

GEN_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
    "required": ["question", "answer"],
}


def gen_prompt(passage: Passage, qtype: str, seed: int) -> str:
    rng = random.Random(seed)
    return _GEN_INSTRUCTIONS.format(
        type_rule=_TYPE_RULES[qtype],
        style_hint=(f"Vary the phrasing: make it {rng.choice(_STYLE_HINTS)}."
                    if qtype != "unanswerable" else
                    "Vary the phrasing across the usual question words."),
        length_hint=(_LENGTH_HINTS[0] if qtype == "unanswerable"
                     else rng.choice(_LENGTH_HINTS)),
        passage=passage.text,
    )


# =============================================================================
# Candidate + format validation (spec page: length & format filters)
# =============================================================================
@dataclass
class Candidate:
    id: str
    source: str
    source_file: str
    type: str
    question: str
    answer: str
    context: str
    context_tokens: int
    llm_judge_score: int | None = None
    llm_judge_verdict: str | None = None
    llm_judge_reason: str | None = None
    drop_reason: str | None = None
    extras: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


_CONTEXT_LEAK = re.compile(
    r"\b(the|this|that|provided|given|above|following)\s+"
    r"(passage|context|document|excerpt|text|filing|snippet)\b", re.I)
_TRUNCATED = re.compile(r"(\.\.\.|…)\s*$")


def parse_generation(raw: str) -> dict | None:
    """Tolerate a stray code fence around otherwise-JSON output."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def validate(qtype: str, question: str, answer: str) -> str | None:
    """Return a drop reason, or None if the pair is well formed."""
    q, a = (question or "").strip(), (answer or "").strip()
    if not q or not a:
        return "empty_field"
    if len(q) < sc.QUESTION_MIN_CHARS:
        return "question_too_short"
    if "?" not in q:
        return "not_a_question"
    if _CONTEXT_LEAK.search(q):
        return "context_leak_in_question"
    if _TRUNCATED.search(a):
        return "answer_truncated"
    if qtype == "unanswerable":
        if _normalize(a) != _normalize(sc.REFUSAL_TEXT):
            return "refusal_text_wrong"
        return None
    if len(a) < sc.ANSWER_MIN_CHARS:
        return "answer_too_short"
    if len(a.split()) > sc.ANSWER_MAX_WORDS * 1.25:
        return "answer_too_long"
    if _normalize(a) == _normalize(sc.REFUSAL_TEXT):
        return "unexpected_refusal"
    return None


# =============================================================================
# Normalization / hashing / n-grams
# =============================================================================
def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return _WS.sub(" ", _PUNCT.sub(" ", text)).strip()


normalize_question = _normalize


def question_hash(question: str) -> str:
    return hashlib.sha1(_normalize(question).encode("utf-8")).hexdigest()


def answer_hash(answer: str) -> str:
    return hashlib.sha1(_normalize(answer).encode("utf-8")).hexdigest()


def ngrams(text: str, n: int = sc.DECONTAM_NGRAM_N) -> set[str]:
    words = _normalize(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


# =============================================================================
# Judge prompt (batched -- one call scores JUDGE_BATCH_SIZE pairs)
# =============================================================================
_JUDGE_INSTRUCTIONS = """You are grading synthetic training pairs for a legal/financial assistant.

For EACH item below, judge the pair against ITS OWN passage only.

Score 1-5 for correctness and grounding:
  5 = answer is fully correct and every fact in it appears in the passage
  4 = correct and grounded, minor wording looseness
  3 = mostly right but adds or omits something material
  2 = partly wrong, or uses facts not in the passage
  1 = wrong, hallucinated, or the "question" is just a restatement of the passage

Also set these booleans:
  grounded    - true if the answer adds NO fact absent from the passage
  real_question - true if it is a genuine question, not a restatement/summary
  refusal_correct - for type "unanswerable": true only if the passage genuinely does
                    NOT answer the question AND the given answer refuses.
                    For other types: true if the answer does not wrongly refuse.

verdict = "keep" only if score >= 4 AND grounded AND real_question AND refusal_correct.
Otherwise "drop".

reason: at most 15 words. No explanation beyond that.

Return JSON only: {"results": [{"idx": 0, "score": 5, "grounded": true, "real_question": true, "refusal_correct": true, "verdict": "keep", "reason": "..."}]}
Return exactly one result object per item, in order.

ITEMS:
"""

JUDGE_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "score": {"type": "integer"},
                    "grounded": {"type": "boolean"},
                    "real_question": {"type": "boolean"},
                    "refusal_correct": {"type": "boolean"},
                    "verdict": {"type": "string", "enum": ["keep", "drop"]},
                    "reason": {"type": "string"},
                },
                "required": ["idx", "score", "grounded", "real_question",
                             "refusal_correct", "verdict", "reason"],
            },
        }
    },
    "required": ["results"],
}


def judge_prompt(batch: list[Candidate], passage_chars: int = sc.JUDGE_PASSAGE_CHARS) -> str:
    blocks = []
    for i, c in enumerate(batch):
        blocks.append(
            f"--- ITEM {i} (type: {c.type}) ---\n"
            f"PASSAGE:\n\"\"\"\n{c.context[:passage_chars]}\n\"\"\"\n"
            f"QUESTION: {c.question}\n"
            f"ANSWER: {c.answer}\n"
        )
    return _JUDGE_INSTRUCTIONS + "\n".join(blocks)


def apply_judgement(cand: Candidate, result: dict) -> Candidate:
    score = int(result.get("score", 0) or 0)
    flags_ok = bool(result.get("grounded")) and bool(result.get("real_question")) \
        and bool(result.get("refusal_correct"))
    verdict = "keep" if (score >= sc.JUDGE_KEEP_SCORE and flags_ok) else "drop"
    cand.llm_judge_score = score
    cand.llm_judge_verdict = verdict
    cand.llm_judge_reason = str(result.get("reason", ""))[:200]
    if verdict == "drop":
        if score < sc.JUDGE_KEEP_SCORE:
            cand.drop_reason = "judge_score"
        elif not result.get("grounded"):
            cand.drop_reason = "judge_ungrounded"
        elif not result.get("real_question"):
            cand.drop_reason = "judge_not_a_question"
        else:
            cand.drop_reason = "judge_refusal_wrong"
    return cand


# =============================================================================
# Chat rendering (SLM_FINE_TUNING.md section 5 -- fixed format)
# =============================================================================
def render_chat(question: str, answer: str, context: str) -> tuple[str, str]:
    """(prompt_part, answer_part). Loss is computed on the answer part only."""
    prompt = (f"<|bos|><|system|>{sc.SYSTEM_PROMPT}"
              f"<|user|>Context:\n{context}\n\nQuestion: {question}"
              f"<|assistant|>")
    return prompt, f"{answer}<|eos|>"


def render_example(cand: Candidate) -> tuple[str, str]:
    return render_chat(cand.question, cand.answer, cand.context)


def encode_example(cand: Candidate, tok) -> tuple[list[int], int] | None:
    """Token ids for the full example plus the index where assistant tokens start.

    Special tokens are inserted as TEXT and encoded by the same BPE (they are
    real vocab entries), so add_special_tokens is off everywhere. Consistent.
    """
    prompt, completion = render_example(cand)
    p_ids = tok.encode(prompt, add_special_tokens=False)
    c_ids = tok.encode(completion, add_special_tokens=False)
    ids = p_ids + c_ids
    if len(ids) > sc.SEQ_LEN:
        return None
    return ids, len(p_ids)


# =============================================================================
# Cosine / greedy near-dup
# =============================================================================
def cosine_dedup(vectors, order: list[int], threshold: float = sc.NEAR_DUP_COSINE) -> list[int]:
    """Greedy: keep a row only if it is below `threshold` against every kept row.

    `vectors` is an L2-normalized (n, d) float32 array; `order` is the priority
    in which rows are considered (best first). Returns kept indices.
    """
    import numpy as np

    kept: list[int] = []
    buf = np.empty((len(order), vectors.shape[1]), dtype=np.float32)
    n = 0
    for i in order:
        v = vectors[i]
        if n and float((buf[:n] @ v).max()) >= threshold:
            continue
        buf[n] = v
        n += 1
        kept.append(i)
    return kept


# =============================================================================
# Accuracy judge (Phase 3): grades GENERATED answers, not training data
# =============================================================================
_ACC_INSTRUCTIONS = """You are grading a small model's answers against a source passage.

For EACH item you are given the PASSAGE, the QUESTION, the GOLD answer written by a
careful reader, and the MODEL answer under test.

Judge the MODEL answer only:
  correct   - true if it conveys the same substantive information as GOLD, or is
              otherwise fully supported by the PASSAGE. Wording may differ freely.
              A missing key figure, a wrong number, name or date makes it false.
  grounded  - true if every fact it asserts appears in the PASSAGE. An answer that
              invents a figure, date or entity is NOT grounded even if it sounds right.
  refusal   - true if the MODEL answer declines to answer (e.g. "the context does not say").
  verdict   - "correct" if correct AND grounded.
              "refused" if the model refused.
              "wrong" otherwise.

For items whose GOLD is itself a refusal, the correct behaviour IS to refuse: mark
correct=true and verdict="correct" when the model refuses, and wrong when it answers.

reason: at most 12 words.

Return JSON only:
{"results":[{"idx":0,"correct":true,"grounded":true,"refusal":false,"verdict":"correct","reason":"..."}]}
Return exactly one result object per item, in order.

ITEMS:
"""

ACC_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "correct": {"type": "boolean"},
                    "grounded": {"type": "boolean"},
                    "refusal": {"type": "boolean"},
                    "verdict": {"type": "string",
                                "enum": ["correct", "refused", "wrong"]},
                    "reason": {"type": "string"},
                },
                "required": ["idx", "correct", "grounded", "refusal", "verdict", "reason"],
            },
        }
    },
    "required": ["results"],
}


def accuracy_prompt(batch: list[dict], passage_chars: int = sc.JUDGE_PASSAGE_CHARS) -> str:
    blocks = []
    for i, r in enumerate(batch):
        blocks.append(
            f"--- ITEM {i} (type: {r['type']}) ---\n"
            f"PASSAGE:\n\"\"\"\n{r['context'][:passage_chars]}\n\"\"\"\n"
            f"QUESTION: {r['question']}\n"
            f"GOLD: {r['gold']}\n"
            f"MODEL: {r['generated']}\n"
        )
    return _ACC_INSTRUCTIONS + "\n".join(blocks)
