#!/usr/bin/env bash
# monitor_training.sh — polls Modal volume every 10 min and prints a status line.
# Exits when training completes (step 15556) or stops progressing for 30+ min.

VOLUME="slm125mLIVE-anand"
METRICS_REMOTE="checkpoints/metrics.jsonl"
METRICS_LOCAL="/tmp/slm125_metrics_monitor.jsonl"
TOTAL_STEPS=15556
POLL_INTERVAL=600   # 10 minutes
MAX_STALL=3         # exit after 3 consecutive polls with no new steps (~30 min stall)

stall=0
last_step=-1

echo "[monitor] started at $(date). polling every ${POLL_INTERVAL}s."

while true; do
    modal volume get "$VOLUME" "$METRICS_REMOTE" "$METRICS_LOCAL" 2>/dev/null

    if [[ ! -f "$METRICS_LOCAL" ]]; then
        echo "[monitor] $(date '+%H:%M') metrics not found yet — waiting"
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Latest training step
    cur_step=$(grep '"train_loss"' "$METRICS_LOCAL" | tail -1 | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d['step'])" 2>/dev/null || echo 0)
    # Latest val_loss
    val=$(grep '"val_loss"' "$METRICS_LOCAL" | tail -1 | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d['val_loss'])" 2>/dev/null || echo "—")
    # Latest train_loss
    tloss=$(grep '"train_loss"' "$METRICS_LOCAL" | tail -1 | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d['train_loss'])" 2>/dev/null || echo "—")
    # Tokens seen
    tokens=$(grep '"train_loss"' "$METRICS_LOCAL" | tail -1 | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('tokens_seen_B','—'))" 2>/dev/null || echo "—")

    pct=$(python3 -c "print(f'{$cur_step/$TOTAL_STEPS*100:.1f}')" 2>/dev/null || echo "?")
    echo "[monitor] $(date '+%H:%M') step=${cur_step}/${TOTAL_STEPS} (${pct}%)  train_loss=${tloss}  val_loss=${val}  tokens=${tokens}B"

    # Done?
    if [[ "$cur_step" -ge "$TOTAL_STEPS" ]]; then
        echo "[monitor] TRAINING COMPLETE at step ${cur_step}. exiting."
        exit 0
    fi

    # Stall detection
    if [[ "$cur_step" -eq "$last_step" ]]; then
        stall=$((stall + 1))
        echo "[monitor] WARNING: no progress for ${stall} consecutive polls (${POLL_INTERVAL}s each)"
        if [[ "$stall" -ge "$MAX_STALL" ]]; then
            echo "[monitor] STALLED — no new steps after $((stall * POLL_INTERVAL / 60)) minutes. exiting."
            exit 1
        fi
    else
        stall=0
    fi

    last_step="$cur_step"
    sleep "$POLL_INTERVAL"
done
