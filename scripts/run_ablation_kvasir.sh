#!/bin/bash
# =============================================================================
# Kvasir 10% ablation runner for AdaPVL
#
# Usage:
#   bash scripts/run_ablation_kvasir.sh <output_dir> <seed>
#
# Example:
#   bash scripts/run_ablation_kvasir.sh output_ablation 1
# =============================================================================

set -e

OUTPUT_DIR=${1:-"output_ablation"}
SEED=${2:-1}

BACKBONES=("adapvl_evaclip" "adapvl_dinov3")
CONFIG_NAMES=(
    "aagf_only"
    "aagf_cmas"
    "full"
    "global_gates"
    "no_cmas"
    "all10_layers"
)

run_case() {
    local backbone=$1
    local run_name=$2
    shift 2

    echo "============================================"
    echo "Backbone: ${backbone}"
    echo "Case: ${run_name}"
    echo "============================================"

    python train.py \
        --config-file configs/Kvasir.yaml \
        --seed ${SEED} \
        --data_percentage 10 \
        --output-dir ${OUTPUT_DIR}/${run_name} \
        MODEL.CLIP_MODEL ${backbone} \
        "$@"

    python test.py \
        --config-file configs/Kvasir.yaml \
        --seed ${SEED} \
        --data_percentage 10 \
        --source_dataset Kvasir_10 \
        --output-dir ${OUTPUT_DIR}/${run_name} \
        MODEL.CLIP_MODEL ${backbone} \
        TEST.USE_LATEST False

    python utils/eval.py \
        --config-file configs/Kvasir.yaml \
        --seed ${SEED} \
        --data_percentage 10 \
        --output-dir ${OUTPUT_DIR}/${run_name} \
        MODEL.CLIP_MODEL ${backbone}
}

for backbone in "${BACKBONES[@]}"; do
    run_case ${backbone} aagf_only \
        MODEL.USE_CMAS False \
        MODEL.USE_MLFA False

    run_case ${backbone} aagf_cmas \
        MODEL.USE_CMAS True \
        MODEL.USE_MLFA False

    run_case ${backbone} full \
        MODEL.USE_CMAS True \
        MODEL.USE_MLFA True

    run_case ${backbone} global_gates \
        MODEL.USE_CMAS True \
        MODEL.USE_MLFA True \
        MODEL.SHARE_GATES True

    run_case ${backbone} no_cmas \
        MODEL.USE_CMAS False \
        MODEL.USE_MLFA True

    run_case ${backbone} all10_layers \
        MODEL.USE_CMAS True \
        MODEL.USE_MLFA True \
        MODEL.MLFA_ALL_LAYERS True
done
