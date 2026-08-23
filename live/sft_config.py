"""Single source of truth for the SFT (instruction fine-tune) run.

Companion to config.py, which owns the pretraining build. Nothing here
retrains a tokenizer or changes the model: the SFT set is encoded with the
SAME 16,384-token BPE that produced /data/tokens.

The cost model is the direction table from SLM_FINE_TUNING.md section 0.1.
Raise COST_LIMIT_USD to spend more; the pair count and the GPU envelope are
derived from it. Code must refuse to launch generate / train when the
projection is over the limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import config

# =============================================================================
# 0. The only knob
# =============================================================================
COST_LIMIT_USD: float = 15.0        # hard ceiling for Phase 1 + Phase 2
COST_LIMIT_FLOOR_USD: float = 10.0  # do not go below this without asking
ABORT_AT_FRACTION: float = 0.95     # stop the run at 95% of the limit

DATASET_FRACTION: float = 0.75      # Gemini generate + judge + embeddings
GPU_FRACTION: float = 0.20          # Modal GPU train + eval + publish
BUFFER_FRACTION: float = 0.05       # retries, Modal CPU, rounding

# Measured unit costs (SLM_FINE_TUNING.md 0.1). Do not switch models.
COST_PER_GEN_CALL_USD: float = 4.3 / 4000      # ~$0.001075
COST_PER_JUDGE_CALL_USD: float = 4.5 / 4000    # ~$0.001125
COST_PER_EMBED_TEXT_USD: float = 0.05 / 13000  # ~$0.0000038

# List prices checked 2026-08-23 (ai.google.dev/gemini-api/docs/pricing), used
# to bill live usage exactly. gemini-3.6-flash promotional rate; it doubles on
# 2027-01-01, so re-check before any rerun after that date.
USD_PER_1M_FLASH_INPUT: float = 0.75
USD_PER_1M_FLASH_OUTPUT: float = 3.75
USD_PER_1M_EMBED_INPUT: float = 0.15

# Measured-shape projection for the chosen model. Token counts come from the
# actual prompts in sft_gen.py; the live tracker overrides these with metered
# usage as soon as the first shard lands.
EST_GEN_INPUT_TOKENS: int = 800       # ~700-BPE passage + instructions
EST_GEN_OUTPUT_TOKENS: int = 150      # {"question": ..., "answer": ...}
EST_JUDGE_INPUT_TOKENS: int = 6_000   # JUDGE_BATCH_SIZE items + rubric
EST_JUDGE_OUTPUT_TOKENS: int = 320    # one small verdict object per item
EST_EMBED_TOKENS_PER_TEXT: int = 60   # questions and answers are short

EMBED_TEXTS_PER_CANDIDATE: int = 3             # question + answer + passage
COST_PER_CANDIDATE_USD: float = (
    COST_PER_GEN_CALL_USD
    + COST_PER_JUDGE_CALL_USD
    + EMBED_TEXTS_PER_CANDIDATE * COST_PER_EMBED_TEXT_USD
)

# Quality band. Never below 2,500; never above 6,000 at $15.
N_CANDIDATES_MIN: int = 2_500
N_CANDIDATES_MAX: int = 6_000

# Worked examples from section 0.2 win over the raw division: the table is the
# plan of record, the formula is only the fallback for other cost limits.
_WORKED_N_CANDIDATES: dict[float, int] = {10.0: 3_400, 15.0: 4_000}

PILOT_CANDIDATES: int = 100         # measure real $/1k before the full spend

# =============================================================================
# 1. Models (fixed -- Pro and thinking mode break the cost table)
# =============================================================================
# SLM_FINE_TUNING.md names gemini-2.5-flash, but Google retired it (404 "no
# longer available", probed 2026-08-23). gemini-3.6-flash is the chosen
# successor: $0.75 in / $3.75 out per 1M (promotional through 2026-12-31).
# Dearer per token than the direction table assumed, but a batched judge more
# than pays the difference back -- see the projection in __main__ below.
GEMINI_GEN_MODEL: str = "gemini-3.6-flash"
GEMINI_JUDGE_MODEL: str = "gemini-3.6-flash"
GEMINI_EMBED_MODEL: str = "gemini-embedding-001"
# Gemini 3.x replaced thinking_budget with thinking_level; "low" is the floor
# and is this API's equivalent of the brief's "thinking off".
GEMINI_THINKING_LEVEL: str = "low"
GEMINI_SECRET_NAME: str = "gemini-api-key"
JUDGE_BATCH_SIZE: int = 8           # batched judge; no per-pair chain of thought

# Fan-out. Kept deliberately modest: the binding constraint is the Gemini
# rate limit, not Modal CPU. ~40 concurrent calls -> well inside tier-1 RPM.
GEN_MAX_CONTAINERS: int = 20
GEN_THREADS_PER_WORKER: int = 2
GEN_MAX_OUTPUT_TOKENS: int = 512
JUDGE_MAX_CONTAINERS: int = 10
JUDGE_THREADS_PER_WORKER: int = 2
JUDGE_MAX_OUTPUT_TOKENS: int = 1_400
EMBED_BATCH_SIZE: int = 100         # gemini-embedding-001 request cap
EMBED_TEXTS_PER_MINUTE: int = 2_400  # paid-tier quota is 3,000 texts/min

# =============================================================================
# 2. Dataset shape
# =============================================================================
KEPT_TARGET_MIN: int = 2_500
KEPT_TARGET_MAX: int = 3_000
EVAL_PAIRS: int = 200

SOURCE_MIX: dict[str, float] = {"case-law": 0.40, "sec": 0.40, "fineweb-edu": 0.20}
TYPE_MIX: dict[str, float] = {"lookup": 0.50, "reasoning": 0.30, "unanswerable": 0.20}

JUDGE_KEEP_SCORE: int = 4           # keep only score >= 4 out of 5
NEAR_DUP_COSINE: float = 0.92       # question-embedding near-dup threshold
DECONTAM_NGRAM_N: int = 13          # eval-vs-train n-gram overlap check
DECONTAM_COSINE: float = 0.88       # eval-vs-train paraphrase threshold (stricter)
ANSWER_MAX_WORDS: int = 120
ANSWER_MIN_CHARS: int = 20
QUESTION_MIN_CHARS: int = 15

# Passage budget so the rendered example fits the 1,024-token context.
SEQ_LEN: int = config.SEQ_LEN
PASSAGE_MAX_TOKENS: int = 700
PASSAGE_MIN_TOKENS: int = 120        # too short -> nothing worth asking about
PASSAGE_MIN_CHARS: int = 500
JUDGE_PASSAGE_CHARS: int = 2_400     # judge sees the passage, trimmed for cost
REFUSAL_TEXT: str = "The provided context does not say."
SYSTEM_PROMPT: str = (
    "You are a legal and financial assistant.\n"
    "Answer only from the provided context.\n"
    "If the context is not enough, say you do not know."
)

# =============================================================================
# 3. Volume layout (new files; never overwrite the pretraining ones)
# =============================================================================
SFT_DIR = f"{config.DATA_ROOT}/sft"
SFT_RAW_DIR = f"{SFT_DIR}/raw"
SFT_JUDGED_DIR = f"{SFT_DIR}/judged"
SFT_KEPT_PATH = f"{SFT_DIR}/kept.jsonl"
SFT_EVAL_PATH = f"{SFT_DIR}/eval.jsonl"
SFT_TOKENS_DIR = f"{SFT_DIR}/tokens"
SFT_TRAIN_TOKENS_DIR = f"{SFT_TOKENS_DIR}/train"
SFT_VAL_TOKENS_DIR = f"{SFT_TOKENS_DIR}/val"
SFT_TOKENS_INDEX = f"{SFT_TOKENS_DIR}/index.json"
SFT_STATS_PATH = f"{SFT_DIR}/stats.json"

# =============================================================================
# 4. Phase 2 planning defaults (proposals only -- STOP GATE B decides)
# =============================================================================
SFT_GPU: str = "L40S"
SFT_GPU_COUNT: int = 1
SFT_GPU_USD_PER_HOUR: float = 1.95
H100_USD_PER_HOUR: float = 3.9492
L40S_BF16_PEAK: float = 362.05e12   # dense bf16 FLOP/s, for MFU

SFT_CKPT_DIR = f"{config.CKPT_DIR}/sft"
SFT_RESUME_PATH = f"{SFT_CKPT_DIR}/ckpt.pt"
SFT_METRICS_PATH = f"{SFT_CKPT_DIR}/metrics.jsonl"
SFT_HF_REPO = f"{config.HF_REPO}-sft"   # never overwrite the base repo


@dataclass(frozen=True)
class SFTTrainConfig:
    """Approved at STOP GATE B on 2026-08-23. An order of magnitude below the
    pretraining LR, and a global batch 8x smaller -- this is instruction tuning
    on ~2.6k examples, not another 8B-token pretrain."""

    seq_len: int = SEQ_LEN
    micro_batch_size: int = 16
    global_batch_tokens: int = 65_536    # 64 windows/step (brief's floor)
    epochs: int = 3
    lr: float = 3e-5
    min_lr: float = 3e-6
    warmup_steps: int = 10
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    log_every_steps: int = 5
    eval_every_steps: int = 20
    ckpt_every_steps: int = 40
    seed: int = 1337


SFT_TRAIN = SFTTrainConfig()


@dataclass(frozen=True)
class Envelopes:
    limit: float
    dataset: float
    gpu: float
    buffer: float


def envelopes(limit: float = COST_LIMIT_USD) -> Envelopes:
    return Envelopes(
        limit=limit,
        dataset=limit * DATASET_FRACTION,
        gpu=limit * GPU_FRACTION,
        buffer=limit * BUFFER_FRACTION,
    )


def n_candidates(limit: float = COST_LIMIT_USD) -> int:
    """Pairs the dataset envelope can buy, clamped to the quality band."""
    worked = _WORKED_N_CANDIDATES.get(round(limit, 2))
    if worked is not None:
        return worked
    derived = math.floor((limit * DATASET_FRACTION) / COST_PER_CANDIDATE_USD)
    return max(N_CANDIDATES_MIN, min(N_CANDIDATES_MAX, derived))


def dataset_cost(n: int) -> dict[str, float]:
    """Projection at the CHOSEN model's list prices and this pipeline's shapes.

    The direction table's flat per-call rates were costed on a model Google has
    since retired, so they are kept only as a reference line.
    """
    gen = n * live_cost(EST_GEN_INPUT_TOKENS, EST_GEN_OUTPUT_TOKENS)
    judge_calls = math.ceil(n / JUDGE_BATCH_SIZE)
    judge = judge_calls * live_cost(EST_JUDGE_INPUT_TOKENS, EST_JUDGE_OUTPUT_TOKENS)
    embed = live_cost(0, 0, n * EMBED_TEXTS_PER_CANDIDATE * EST_EMBED_TOKENS_PER_TEXT)
    table = n * (COST_PER_GEN_CALL_USD + COST_PER_JUDGE_CALL_USD)
    return {"gen": gen, "judge": judge, "embed": embed, "judge_calls": judge_calls,
            "total": gen + judge + embed, "direction_table_ref": table}


def live_cost(input_tokens: int, output_tokens: int, embed_tokens: int = 0) -> float:
    """Exact bill from metered token counts, at list prices. Used by the tracker."""
    return (input_tokens * USD_PER_1M_FLASH_INPUT / 1e6
            + output_tokens * USD_PER_1M_FLASH_OUTPUT / 1e6
            + embed_tokens * USD_PER_1M_EMBED_INPUT / 1e6)


def check_dataset_budget(n: int, limit: float = COST_LIMIT_USD) -> tuple[bool, str]:
    """GO / NO-GO for a generate run of n candidates."""
    env = envelopes(limit)
    cost = dataset_cost(n)["total"]
    ok = cost <= env.dataset
    verdict = "GO" if ok else "NO-GO"
    return ok, (f"{verdict}: {n:,} candidates -> ${cost:.2f} dataset vs "
                f"${env.dataset:.2f} envelope (limit ${limit:.2f})")


def fit_n_to_budget(limit: float = COST_LIMIT_USD) -> int:
    """Largest n that still fits the dataset envelope, honouring the band."""
    n = n_candidates(limit)
    while n > N_CANDIDATES_MIN and dataset_cost(n)["total"] > envelopes(limit).dataset:
        n -= 100
    return n


def gpu_budget(phase1_actual_usd: float, limit: float = COST_LIMIT_USD) -> float:
    """Money left for Phase 2 after Phase 1 actuals and the buffer."""
    return limit - phase1_actual_usd - (limit * BUFFER_FRACTION)


def gpu_cost(hours: float, usd_per_hour: float = SFT_GPU_USD_PER_HOUR,
             gpu_count: int = SFT_GPU_COUNT) -> float:
    return hours * usd_per_hour * gpu_count


if __name__ == "__main__":
    env = envelopes()
    n = n_candidates()
    cost = dataset_cost(n)
    pilot = dataset_cost(PILOT_CANDIDATES)
    print(f"COST_LIMIT_USD           ${env.limit:.2f}")
    print(f"  dataset envelope (75%) ${env.dataset:.2f}")
    print(f"  gpu envelope     (20%) ${env.gpu:.2f}")
    print(f"  buffer            (5%) ${env.buffer:.2f}")
    print(f"  abort at 95%           ${env.limit * ABORT_AT_FRACTION:.2f}")
    print()
    print(f"cost per candidate       ${COST_PER_CANDIDATE_USD:.5f} "
          f"(${COST_PER_CANDIDATE_USD * 1000:.2f} / 1k pairs)")
    print(f"model                    {GEMINI_GEN_MODEL} "
          f"(${USD_PER_1M_FLASH_INPUT}/1M in, ${USD_PER_1M_FLASH_OUTPUT}/1M out)")
    print(f"n_candidates             {n:,}")
    print(f"  generate               ${cost['gen']:.2f}  ({n:,} calls)")
    print(f"  judge                  ${cost['judge']:.2f}  "
          f"({cost['judge_calls']:,.0f} batched calls of {JUDGE_BATCH_SIZE})")
    print(f"  embed                  ${cost['embed']:.2f}")
    print(f"  dataset total          ${cost['total']:.2f}")
    print(f"  (direction-table ref   ${cost['direction_table_ref']:.2f} on the "
          f"retired model)")
    print(f"pilot ({PILOT_CANDIDATES} candidates)     ${pilot['total']:.2f}")
    print()
    print(check_dataset_budget(n)[1])
    print(f"kept target              {KEPT_TARGET_MIN:,}-{KEPT_TARGET_MAX:,} "
          f"(+{EVAL_PAIRS} held-out eval)")
