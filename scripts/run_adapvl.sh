#!/bin/bash
# =============================================================================
# AdaPVL Experiment Runner
#
# Usage:
#   bash scripts/run_adapvl.sh <output_dir> <seed>
#
# Example:
#   bash scripts/run_adapvl.sh output_adapvl_v2 1
#
# This runs:
#   Optional Phase 1: Kvasir 10% on both backbones (quick validation)
#   Phase 2: Full data efficiency (6 datasets x 4 percentages)
#   Phase 3: Domain generalization (4 domains)
# Environment overrides:
#   BACKBONES="adapvl_evaclip adapvl_dinov3_siglip"
# =============================================================================

set -e

OUTPUT_DIR=${1:-"output_adapvl"}
SEED=${2:-1}
RUN_PHASE1=${RUN_PHASE1:-0}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-}
TEST_USE_LATEST=${TEST_USE_LATEST:-false}
CHECKPOINT_TYPE="best_dice"
if [ "${TEST_USE_LATEST}" = "true" ]; then
    CHECKPOINT_TYPE="latest"
fi

echo "============================================"
echo "AdaPVL Experiments"
echo "Output: ${OUTPUT_DIR}, Seed: ${SEED}"
echo "Quick validation (Phase 1): ${RUN_PHASE1}"
echo "Test checkpoint type: ${CHECKPOINT_TYPE}"
if [ -n "${TRAIN_BATCH_SIZE}" ]; then
    echo "Train batch size override: ${TRAIN_BATCH_SIZE}"
else
    echo "Train batch size override: config default"
fi
echo "============================================"

# Datasets for data efficiency evaluation
# Original MedCLIPSeg data-efficiency protocol uses 6 datasets
# at 10/25/50/100% training data.
DATASETS=("BUSI" "BTMRI" "ISIC" "Kvasir" "Covid19" "EUS")
PERCENTAGES=(10 25 50 100)

# Backbone variants to test
if [ -n "${BACKBONES:-}" ]; then
    read -r -a BACKBONES <<< "${BACKBONES}"
else
    BACKBONES=("adapvl_evaclip" "adapvl_dinov3")
fi

TRAIN_EXTRA_OPTS=()
if [ -n "${TRAIN_BATCH_SIZE}" ]; then
    TRAIN_EXTRA_OPTS+=(TRAIN.BATCH_SIZE "${TRAIN_BATCH_SIZE}")
fi

TEST_EXTRA_OPTS=(TEST.USE_LATEST "${TEST_USE_LATEST}")

if [ "${RUN_PHASE1}" = "1" ]; then
    # =============================================
    # Phase 1: Quick validation on Kvasir 10%
    # =============================================
    echo ""
    echo "========== Phase 1: Quick Validation (Kvasir 10%) =========="
    for BB in "${BACKBONES[@]}"; do
        echo "--- Training ${BB} on Kvasir 10% ---"
        python train.py \
            --config-file configs/Kvasir.yaml \
            --seed ${SEED} \
            --data_percentage 10 \
            --output-dir ${OUTPUT_DIR} \
            MODEL.CLIP_MODEL ${BB} \
            "${TRAIN_EXTRA_OPTS[@]}"

        echo "--- Testing ${BB} on Kvasir 10% ---"
        python test.py \
            --config-file configs/Kvasir.yaml \
            --seed ${SEED} \
            --data_percentage 10 \
            --source_dataset Kvasir_10 \
            --output-dir ${OUTPUT_DIR} \
            MODEL.CLIP_MODEL ${BB} \
            "${TEST_EXTRA_OPTS[@]}"

        echo "--- Evaluating ${BB} on Kvasir 10% ---"
        python utils/eval.py \
            --config-file configs/Kvasir.yaml \
            --seed ${SEED} \
            --data_percentage 10 \
            --output-dir ${OUTPUT_DIR} \
            MODEL.CLIP_MODEL ${BB}
    done

    echo ""
    echo "========== Phase 1 Complete =========="
    echo "Check results in ${OUTPUT_DIR}/Kvasir_10/seg_results/seed${SEED}/"
    echo "Gate trajectories are logged in trained_models/seed${SEED}/log.txt"
    echo ""
fi

# =============================================
# Phase 2: Full Data Efficiency
# =============================================
echo "========== Phase 2: Data Efficiency (6 datasets x 4%) =========="
for BB in "${BACKBONES[@]}"; do
    for DS in "${DATASETS[@]}"; do
        for PCT in "${PERCENTAGES[@]}"; do
            DS_SUFFIX="${DS}"
            if [ "${PCT}" -ne 100 ]; then
                DS_SUFFIX="${DS}_${PCT}"
            fi

            echo "--- ${BB} | ${DS} ${PCT}% ---"
            python train.py \
                --config-file configs/${DS}.yaml \
                --seed ${SEED} \
                --data_percentage ${PCT} \
                --output-dir ${OUTPUT_DIR} \
                MODEL.CLIP_MODEL ${BB} \
                "${TRAIN_EXTRA_OPTS[@]}"

            python test.py \
                --config-file configs/${DS}.yaml \
                --seed ${SEED} \
                --data_percentage ${PCT} \
                --source_dataset ${DS_SUFFIX} \
                --output-dir ${OUTPUT_DIR} \
                MODEL.CLIP_MODEL ${BB} \
                "${TEST_EXTRA_OPTS[@]}"

            python utils/eval.py \
                --config-file configs/${DS}.yaml \
                --seed ${SEED} \
                --data_percentage ${PCT} \
                --output-dir ${OUTPUT_DIR} \
                MODEL.CLIP_MODEL ${BB}
        done
    done
done

echo ""
echo "========== Phase 2 Complete =========="

# =============================================
# Phase 3: Domain Generalization
# =============================================
echo "========== Phase 3: Domain Generalization =========="

# BUS domain: BUSI -> BUSBRA, BUSUC, BUID, UDIAT
BUS_TARGETS=("BUSBRA" "BUSUC" "BUID" "UDIAT")
# ENDO domain: Kvasir -> ColonDB, ClinicDB, CVC300, BKAI
ENDO_TARGETS=("ColonDB" "ClinicDB" "CVC300" "BKAI")
# DERM domain: ISIC -> UWaterlooSkinCancer
DERM_TARGETS=("UWaterlooSkinCancer")
# BRAIN domain: BTMRI -> BRISC
BRAIN_TARGETS=("BRISC")

for BB in "${BACKBONES[@]}"; do
    # Models are already trained on 100% data from Phase 2.
    # Just need to test on OOD targets.

    echo "--- ${BB} | BUS domain (BUSI -> targets) ---"
    for TGT in "${BUS_TARGETS[@]}"; do
        python test.py \
            --config-file configs/${TGT}.yaml \
            --seed ${SEED} \
            --source_dataset BUSI \
            --output-dir ${OUTPUT_DIR} \
            MODEL.CLIP_MODEL ${BB} \
            "${TEST_EXTRA_OPTS[@]}"

        python utils/eval.py \
            --config-file configs/${TGT}.yaml \
            --seed ${SEED} \
            --output-dir ${OUTPUT_DIR} \
            MODEL.CLIP_MODEL ${BB}
    done

    echo "--- ${BB} | ENDO domain (Kvasir -> targets) ---"
    for TGT in "${ENDO_TARGETS[@]}"; do
        python test.py \
            --config-file configs/${TGT}.yaml \
            --seed ${SEED} \
            --source_dataset Kvasir \
            --output-dir ${OUTPUT_DIR} \
            MODEL.CLIP_MODEL ${BB} \
            "${TEST_EXTRA_OPTS[@]}"

        python utils/eval.py \
            --config-file configs/${TGT}.yaml \
            --seed ${SEED} \
            --output-dir ${OUTPUT_DIR} \
            MODEL.CLIP_MODEL ${BB}
    done

    echo "--- ${BB} | DERM domain (ISIC -> targets) ---"
    for TGT in "${DERM_TARGETS[@]}"; do
        python test.py \
            --config-file configs/${TGT}.yaml \
            --seed ${SEED} \
            --source_dataset ISIC \
            --output-dir ${OUTPUT_DIR} \
            MODEL.CLIP_MODEL ${BB} \
            "${TEST_EXTRA_OPTS[@]}"

        python utils/eval.py \
            --config-file configs/${TGT}.yaml \
            --seed ${SEED} \
            --output-dir ${OUTPUT_DIR} \
            MODEL.CLIP_MODEL ${BB}
    done

    echo "--- ${BB} | BRAIN domain (BTMRI -> targets) ---"
    for TGT in "${BRAIN_TARGETS[@]}"; do
        python test.py \
            --config-file configs/${TGT}.yaml \
            --seed ${SEED} \
            --source_dataset BTMRI \
            --output-dir ${OUTPUT_DIR} \
            MODEL.CLIP_MODEL ${BB} \
            "${TEST_EXTRA_OPTS[@]}"

        python utils/eval.py \
            --config-file configs/${TGT}.yaml \
            --seed ${SEED} \
            --output-dir ${OUTPUT_DIR} \
            MODEL.CLIP_MODEL ${BB}
    done
done

echo ""
echo "============================================"
echo "All AdaPVL experiments complete!"
echo "Results: ${OUTPUT_DIR}/"
echo "============================================"
