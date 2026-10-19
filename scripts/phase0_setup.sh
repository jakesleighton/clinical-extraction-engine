#!/usr/bin/env bash
# Phase 0 on one machine, start to finish. Safe to re-run: every step is
# idempotent (existing env is updated, existing downloads and an existing
# working set are kept).
#
#   bash scripts/phase0_setup.sh               # everything, then the smoke run
#   bash scripts/phase0_setup.sh --no-ollama   # stop before the model-server steps
#   MODEL=qwen2.5:14b bash scripts/phase0_setup.sh      # different smoke-test model
#   SMOKE_CASE=TCGA-XX-1234 bash scripts/phase0_setup.sh  # smoke-test a specific case
#   WORKING_SET_N=10 bash scripts/phase0_setup.sh        # size of the initial random draw
#
# Steps: miniforge env -> pytest -> stage TCGA-PRAD reports -> gold labels ->
# working set (random draw if none yet) -> install/start Ollama -> pull
# model -> extract one report.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME="cee"
MODEL="${MODEL:-qwen2.5:7b}"
WORKING_SET="data/tcga_prad/working_set.txt"
OLLAMA="${OLLAMA_HOST:-http://localhost:11434}"
case "$OLLAMA" in http://*|https://*) ;; *) OLLAMA="http://$OLLAMA" ;; esac

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

step "1/7 conda environment: $ENV_NAME (environment.yml)"
if command -v mamba >/dev/null 2>&1; then CONDA=mamba
elif command -v conda >/dev/null 2>&1; then CONDA=conda
else echo "miniforge not found; install it from https://github.com/conda-forge/miniforge and re-run"; exit 1
fi
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    "$CONDA" env update -n "$ENV_NAME" -f environment.yml --prune
else
    "$CONDA" env create -n "$ENV_NAME" -f environment.yml
fi
conda activate "$ENV_NAME"
python --version

step "2/7 tests"
pytest

step "3/7 stage TCGA-PRAD reports (download once, filter by GDC case list)"
python -m scripts.fetch_tcga_reports

step "4/7 gold labels and mismatch report"
python -m scripts.join_gold_labels

step "5/7 working set ($WORKING_SET)"
if [[ -s "$WORKING_SET" ]]; then
    echo "keeping the existing working set (pick/add/remove with: python -m scripts.working_set ...)"
    python -m scripts.working_set show
else
    echo "no working set yet: drawing ${WORKING_SET_N:-20} cases at random for the smoke test"
    python -m scripts.working_set random --n "${WORKING_SET_N:-20}"
fi

if [[ "${1:-}" == "--no-ollama" ]]; then
    echo; echo "stopping before Ollama (--no-ollama)"; exit 0
fi

step "6/7 Ollama server + model $MODEL"
if ! command -v ollama >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1; then brew install ollama
    else echo "ollama not found and no Homebrew; install it from https://ollama.com/download and re-run"; exit 1
    fi
fi
if ! curl -sf "$OLLAMA/api/tags" >/dev/null; then
    if command -v brew >/dev/null 2>&1 && brew list --formula ollama >/dev/null 2>&1; then
        brew services start ollama
    else
        nohup ollama serve >/tmp/ollama.log 2>&1 &
    fi
    for _ in $(seq 1 30); do
        curl -sf "$OLLAMA/api/tags" >/dev/null && break
        sleep 1
    done
    curl -sf "$OLLAMA/api/tags" >/dev/null || { echo "Ollama did not come up at $OLLAMA"; exit 1; }
fi
ollama pull "$MODEL"

step "7/7 smoke run: one report through engine.extract"
CASE="${SMOKE_CASE:-}"
if [[ -z "$CASE" && -s "$WORKING_SET" ]]; then
    CASE="$(grep -v '^[[:space:]]*#' "$WORKING_SET" | sed 's/#.*//' | awk 'NF {print $1; exit}')"
fi
if [[ -z "$CASE" ]]; then
    CASE="$(ls data/tcga_prad/reports | head -n 1)"; CASE="${CASE%.txt}"
fi
echo "case $CASE (SMOKE_CASE=... picks another; the first line of $WORKING_SET is the default)"
python -m engine.extract "data/tcga_prad/reports/$CASE.txt" \
    --pack packs/prostate_pathology --model "$MODEL"

echo
echo "done. Paste this terminal's output (from '=== 2/7 tests' down) into the thread."
