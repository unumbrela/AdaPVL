#!/bin/bash
# =============================================================================
# Run ALL experiments with EVA02-CLIP backbone
# Reproduces the original paper's full evaluation: Data Efficiency + Domain Generalization
#
# Usage:
#   bash scripts/run_evaclip_all.sh [OUTPUT_DIR] [SEED]
#   bash scripts/run_evaclip_all.sh output_evaclip 1
#
# This script uses the SAME hyperparameters as the original paper (batch_size=24,
# epochs=100, lr=3e-4, etc.), only changing MODEL.CLIP_MODEL to "evaclip".
# =============================================================================

OUTPUT=${1:-"output_evaclip"}
SEED=${2:-1}
CLIP_MODEL="evaclip"

echo "============================================================"
echo " MedCLIPSeg + EVA02-CLIP Full Evaluation"
echo " Output: ${OUTPUT}  |  Seed: ${SEED}"
echo "============================================================"

# =============================================================================
# PART 1: Data Efficiency Evaluation
# 6 datasets x 4 data percentages = 24 experiments
# =============================================================================
echo ""
echo "============================================================"
echo " PART 1: Data Efficiency Evaluation"
echo "============================================================"

# 6 datasets with 10/25/50/100% prompt subsets
EFFICIENCY_DATASETS=("BUSI" "BTMRI" "ISIC" "Kvasir" "Covid19" "EUS")
DATA_PERCENTAGES=(10 25 50 100)

for DATASET in "${EFFICIENCY_DATASETS[@]}"; do
    for PCT in "${DATA_PERCENTAGES[@]}"; do
        echo ""
        echo "------------------------------------------------------------"
        echo " [Efficiency] ${DATASET} | ${PCT}% data | seed=${SEED}"
        echo "------------------------------------------------------------"

        python train.py \
            --config-file configs/${DATASET}.yaml \
            --output-dir ${OUTPUT} \
            --data_percentage ${PCT} \
            --seed ${SEED} \
            MODEL.CLIP_MODEL ${CLIP_MODEL}

        python test.py \
            --config-file configs/${DATASET}.yaml \
            --output-dir ${OUTPUT} \
            --source_dataset ${DATASET} \
            --data_percentage ${PCT} \
            --seed ${SEED} \
            MODEL.CLIP_MODEL ${CLIP_MODEL}

        python utils/eval.py \
            --config-file configs/${DATASET}.yaml \
            --output-dir ${OUTPUT} \
            --data_percentage ${PCT} \
            --seed ${SEED} \
            MODEL.CLIP_MODEL ${CLIP_MODEL}
    done
done

# =============================================================================
# PART 2: Domain Generalization Evaluation
# 4 domains: BUS(5), ENDO(5), DERM(2), BRAIN(2) = 4 training + 12 OOD tests
# =============================================================================
echo ""
echo "============================================================"
echo " PART 2: Domain Generalization Evaluation"
echo "============================================================"

# Domain definitions (first element = source/ID, rest = target/OOD)
declare -A DOMAINS
DOMAINS[BUS]="BUSI BUSBRA BUSUC BUID UDIAT"
DOMAINS[ENDO]="Kvasir ColonDB ClinicDB CVC300 BKAI"
DOMAINS[DERM]="ISIC UWaterlooSkinCancer"
DOMAINS[BRAIN]="BTMRI BRISC"

for DOMAIN in BUS ENDO DERM BRAIN; do
    read -ra DS_LIST <<< "${DOMAINS[$DOMAIN]}"
    SOURCE=${DS_LIST[0]}

    echo ""
    echo "============================================================"
    echo " [Domain Gen] ${DOMAIN}: Source=${SOURCE}"
    echo "   Targets: ${DS_LIST[@]:1}"
    echo "============================================================"

    # Train on source dataset (100% data) — skip if already trained in Part 1
    CKPT_DIR="${OUTPUT}/${SOURCE}/trained_models/seed${SEED}"
    if [ -f "${CKPT_DIR}/MedCLIPSeg_${CLIP_MODEL}_ViT-B-16_best_dice.pth" ]; then
        echo " -> Skipping training (checkpoint exists from Part 1)"
    else
        python train.py \
            --config-file configs/${SOURCE}.yaml \
            --output-dir ${OUTPUT} \
            --seed ${SEED} \
            MODEL.CLIP_MODEL ${CLIP_MODEL}
    fi

    # Evaluate on source (in-domain)
    echo " -> Eval SOURCE=${SOURCE} (in-domain)"
    python test.py \
        --config-file configs/${SOURCE}.yaml \
        --output-dir ${OUTPUT} \
        --source_dataset ${SOURCE} \
        --seed ${SEED} \
        MODEL.CLIP_MODEL ${CLIP_MODEL}

    python utils/eval.py \
        --config-file configs/${SOURCE}.yaml \
        --output-dir ${OUTPUT} \
        --seed ${SEED} \
        MODEL.CLIP_MODEL ${CLIP_MODEL}

    # Evaluate on each target (out-of-domain)
    for TARGET in "${DS_LIST[@]:1}"; do
        echo " -> Eval SOURCE=${SOURCE} -> TARGET=${TARGET} (OOD)"
        python test.py \
            --config-file configs/${TARGET}.yaml \
            --output-dir ${OUTPUT} \
            --source_dataset ${SOURCE} \
            --seed ${SEED} \
            MODEL.CLIP_MODEL ${CLIP_MODEL}

        python utils/eval.py \
            --config-file configs/${TARGET}.yaml \
            --output-dir ${OUTPUT} \
            --seed ${SEED} \
            MODEL.CLIP_MODEL ${CLIP_MODEL}
    done
done

echo ""
echo "============================================================"
echo " ALL EXPERIMENTS COMPLETE"
echo " Results in: ${OUTPUT}/"
echo "============================================================"
