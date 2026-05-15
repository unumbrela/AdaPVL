#!/bin/bash
# Quick run script for Kvasir with 10% data, adapted for 16GB VRAM
# Usage: bash run_kvasir_quick.sh

set -e

OUTPUT=output
DATASET=Kvasir
DATA_PERCENTAGE=10
SEED=666
BATCH_SIZE=8

echo "========================================="
echo "Step 1: Training on Kvasir (10% data, BS=${BATCH_SIZE})"
echo "========================================="
python train.py --config-file configs/Kvasir.yaml \
    --output-dir ${OUTPUT} \
    --data_percentage ${DATA_PERCENTAGE} \
    --seed ${SEED} \
    TRAIN.BATCH_SIZE ${BATCH_SIZE} \
    TRAIN.NUM_EPOCHS 5

echo "========================================="
echo "Step 2: Testing (inference + segmentation)"
echo "========================================="
python test.py --config-file configs/Kvasir.yaml \
    --output-dir ${OUTPUT} \
    --source_dataset ${DATASET} \
    --data_percentage ${DATA_PERCENTAGE} \
    --seed ${SEED} \
    TRAIN.BATCH_SIZE ${BATCH_SIZE} \
    TEST.NUM_SAMPLES 5

echo "========================================="
echo "Step 3: Evaluation (DSC & NSD metrics)"
echo "========================================="
python utils/eval.py --config-file configs/Kvasir.yaml \
    --output-dir ${OUTPUT} \
    --data_percentage ${DATA_PERCENTAGE} \
    --seed ${SEED}

echo "========================================="
echo "All done! Full pipeline completed."
echo "========================================="
