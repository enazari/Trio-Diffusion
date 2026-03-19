#!/usr/bin/env bash
set -e

# ─── Configuration ───────────────────────────────────────────────────────────
TRAIN_SESSION="sessions/tokens_positioned_dino_large_mar16-20:03"
SIZE="--size 512 512"
GPU=(0)                       # CUDA_VISIBLE_DEVICES

# ─── Enable/disable methods ───────────────────────────────────────────────────
# Free generation — no conditioning. Model generates purely from noise,
# re-encoding the canvas progressively at each patch step.
RUN_PROGRESSIVE=true

# Image → global context, fixed output size. Encodes a seed image once with
# the backbone and uses that fixed embedding as global conditioning for all
# patches. Output is at --size regardless of seed image dimensions.
RUN_GLOBAL_CTX=true

# Text → global context (CLIP only). Encodes a text prompt with CLIP and uses
# it as fixed global conditioning. No image involved.
RUN_TEXT=false

# Image → global context, output matches seed image dimensions. Same as
# RUN_GLOBAL_CTX but output size is inferred from the seed image.
RUN_GLOBAL_CTX_FIT=false

# Image → global + local context. Backbone encodes the seed image for global
# conditioning, AND the actual image patches (TL, TR, BL) are used as local
# context at each step instead of the generated canvas. Closest to inpainting.
RUN_FULL_CTX=false

# ─── Seeds per method ─────────────────────────────────────────────────────────
SEEDS_PROGRESSIVE=(1 2 3 4 5 6 7 8 9 10 11 12 13)
SEEDS_GLOBAL_CTX=(1 2 3 4)
SEEDS_TEXT=(1 2 3 4)
SEEDS_GLOBAL_CTX_FIT=(1 2 3 4)
SEEDS_FULL_CTX=(1)

# ─── Paths (relative to project root) ────────────────────────────────────────
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source "${CONDA_PREFIX:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate trio

SEED_DIR="generation/seeds"
GEN="python scripts/generate.py $TRAIN_SESSION"

# Session name: customize or auto-generate from timestamp
SESSION_NAME="${1:-gen_$(date +%Y%m%d-%H%M)}"
OUT="generation/sessions/$SESSION_NAME/images"
LOG="generation/sessions/$SESSION_NAME/run.log"

mkdir -p "$OUT"

# ─── Log run config ──────────────────────────────────────────────────────────
cat > "$LOG" <<EOF
train_session: $TRAIN_SESSION
size: $SIZE
gpu: $GPU
started: $(date -Iseconds)
EOF
echo "Session: generation/sessions/$SESSION_NAME/"

# ─── Job queue (skip existing) ───────────────────────────────────────────────
JOBS=()

add() {
  local outfile="$1"; shift
  if [ -f "$outfile" ]; then
    echo "SKIP (exists): $outfile"
    return
  fi
  JOBS+=("$outfile|$*")
}

# --- Progressive (no seed) ---
if $RUN_PROGRESSIVE; then
  for S in "${SEEDS_PROGRESSIVE[@]}"; do
    add "$OUT/progressive_s${S}.png" $GEN $SIZE --torch-seed $S
  done
fi

# --- Backbone detection ---
BACKBONE=$(grep -m1 'name:' "$ROOT/$TRAIN_SESSION/config.yaml" 2>/dev/null | awk '{print $2}')

# --- Fixed global context (--fixed-global-context, fixed size, requires backbone) ---
if $RUN_GLOBAL_CTX; then
  if [ "$BACKBONE" != "none" ] && [ -n "$BACKBONE" ]; then
    GLOBAL_CTX_IMAGES=(umbrella flower leaf cat trunk tulip purple blurred)
    for S in "${SEEDS_GLOBAL_CTX[@]}"; do
      for NAME in "${GLOBAL_CTX_IMAGES[@]}"; do
        add "$OUT/global_ctx_${NAME}_s${S}.png" $GEN $SIZE --fixed-global-context "$SEED_DIR/omid-${NAME}.jpg" --torch-seed $S
      done
    done
  else
    echo "Skipping fixed-global-context jobs (backbone: ${BACKBONE:-unknown}, requires backbone)"
  fi
fi

# --- Text seeds (CLIP only) ---
if $RUN_TEXT && [ "$BACKBONE" = "clip" ]; then
  declare -A TEXTS
  TEXTS[sky]="sky"
  TEXTS[building]="building"
  TEXTS[cat]="cat"
  TEXTS[red]="red"
  TEXTS[blue]="blue"
  TEXTS[sunset_lake]="a golden sunset reflecting on a calm lake surrounded by mountains"
  TEXTS[rainy_street]="a rainy city street at night with glowing neon signs and wet pavement"
  TEXTS[forest_fog]="a dense misty forest with sunlight streaming through tall pine trees"
  TEXTS[flower_field]="a vast field of colorful wildflowers under a bright blue sky"
  TEXTS[deep_orange]="a burning orange desert with amber sand dunes under a tangerine sky"
  TEXTS[lush_green]="a dense emerald jungle with bright green ferns and lime moss on every surface"
  TEXTS[vivid_purple]="a field of lavender and violet flowers beneath a deep purple twilight sky"
  TEXTS[bright_yellow]="a golden wheat field glowing in warm yellow sunlight with sunflowers everywhere"
  TEXT_KEYS=(sky building cat red blue sunset_lake rainy_street forest_fog flower_field deep_orange lush_green vivid_purple bright_yellow)
  for S in "${SEEDS_TEXT[@]}"; do
    for K in "${TEXT_KEYS[@]}"; do
      add "$OUT/text_${K}_s${S}.png" $GEN $SIZE --text-seed "${TEXTS[$K]}" --torch-seed $S
    done
  done
else
  $RUN_TEXT || true
  [ "$BACKBONE" != "clip" ] && echo "Skipping text seeds (backbone: ${BACKBONE:-unknown}, requires clip)"
fi

# --- Fixed global context fit (--fixed-global-context-fit, output matches image dimensions, requires backbone) ---
if $RUN_GLOBAL_CTX_FIT; then
  if [ "$BACKBONE" != "none" ] && [ -n "$BACKBONE" ]; then
    GLOBAL_CTX_FIT_IMAGES=()
    for S in "${SEEDS_GLOBAL_CTX_FIT[@]}"; do
      for NAME in "${GLOBAL_CTX_FIT_IMAGES[@]}"; do
        add "$OUT/global_ctx_fit_${NAME}_s${S}.png" $GEN --fixed-global-context-fit "$SEED_DIR/omid-${NAME}.jpg" --torch-seed $S
      done
    done
  else
    echo "Skipping fixed-global-context-fit jobs (backbone: ${BACKBONE:-unknown}, requires backbone)"
  fi
fi

# --- Fixed full context (--fixed-full-context, global backbone + local image patches) ---
if $RUN_FULL_CTX; then
  if [ "$BACKBONE" != "none" ] && [ -n "$BACKBONE" ]; then
    FULL_CTX_IMAGES=(umbrella flower leaf cat trunk tulip)
    for S in "${SEEDS_FULL_CTX[@]}"; do
      for NAME in "${FULL_CTX_IMAGES[@]}"; do
        add "$OUT/full_ctx_${NAME}_s${S}.png" $GEN --fixed-full-context "$SEED_DIR/omid-${NAME}.jpg" --torch-seed $S
      done
    done
  else
    echo "Skipping fixed-full-context jobs (backbone: ${BACKBONE:-unknown}, requires backbone)"
  fi
fi

# ─── Run jobs ─────────────────────────────────────────────────────────────────
echo "=== ${#JOBS[@]} jobs to run (skipped existing), GPU $GPU ==="

for ((i=0; i<${#JOBS[@]}; i++)); do
  IFS='|' read -r out cmd <<< "${JOBS[$i]}"
  echo "[$((i+1))/${#JOBS[@]}] $out"
  CUDA_VISIBLE_DEVICES=$GPU $cmd --output "$out"
done

echo "finished: $(date -Iseconds)" >> "$LOG"
echo "=== Done. Results in $OUT/ ==="
