#!/bin/bash
# Run all model variants on a dataset for comparison
# Usage: bash scripts/run_all_models.sh [OUTPUT] [DATASET] [DATA_PERCENTAGE] [SEED] [BATCH_SIZE] [EPOCHS] [NUM_SAMPLES]

set -e

OUTPUT=${1:-output}
DATASET=${2:-Kvasir}
DATA_PERCENTAGE=${3:-10}
SEED=${4:-666}
BATCH_SIZE=${5:-8}
EPOCHS=${6:-5}
NUM_SAMPLES=${7:-5}

CONFIG=${DATASET}

MODELS=("unimedclip" "evaclip" "siglip" "dinov2" "dinov3" "dinov3_siglip")

for MODEL in "${MODELS[@]}"; do
    echo "========================================="
    echo "Model: ${MODEL} | Dataset: ${DATASET} | Data: ${DATA_PERCENTAGE}%"
    echo "========================================="

    echo "--- Training ---"
    python train.py --config-file configs/${CONFIG}.yaml \
        --output-dir ${OUTPUT} \
        --data_percentage ${DATA_PERCENTAGE} \
        --seed ${SEED} \
        TRAIN.BATCH_SIZE ${BATCH_SIZE} \
        TRAIN.NUM_EPOCHS ${EPOCHS} \
        MODEL.CLIP_MODEL ${MODEL}

    echo "--- Testing ---"
    python test.py --config-file configs/${CONFIG}.yaml \
        --output-dir ${OUTPUT} \
        --source_dataset ${DATASET} \
        --data_percentage ${DATA_PERCENTAGE} \
        --seed ${SEED} \
        TRAIN.BATCH_SIZE ${BATCH_SIZE} \
        TEST.NUM_SAMPLES ${NUM_SAMPLES} \
        MODEL.CLIP_MODEL ${MODEL}

    echo "--- Evaluation ---"
    python utils/eval.py --config-file configs/${CONFIG}.yaml \
        --output-dir ${OUTPUT} \
        --data_percentage ${DATA_PERCENTAGE} \
        --seed ${SEED} \
        MODEL.CLIP_MODEL ${MODEL}

    echo "${MODEL} done!"
    echo ""
done

echo "========================================="
echo "All models completed! Running comparison..."
echo "========================================="
python scripts/compare_results.py --output-dir ${OUTPUT} --dataset ${DATASET} --data_percentage ${DATA_PERCENTAGE} --seed ${SEED}
