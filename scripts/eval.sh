#!/bin/bash
# BLIP-2 Q-Former Evaluation Script
# Evaluates the trained model on Flickr8k test images
#
# Usage:
#   ./scripts/eval.sh                    # Run with defaults
#   ./scripts/eval.sh --skip_retrieval   # Skip retrieval metrics (faster)

# Exit on error
set -e

# Default values
CHECKPOINT="checkpoints/best_model.pt"
DATA_ROOT="data/Flickr8k_Dataset"
OUTPUT_FILE="eval_results.json"
BATCH_SIZE=32
SPLIT="test"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --skip_retrieval)
            SKIP_RETRIEVAL="--skip_retrieval"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "BLIP-2 Q-Former Evaluation"
echo "=============================================="
echo "Checkpoint: ${CHECKPOINT}"
echo "Data Root: ${DATA_ROOT}"
echo "Split: ${SPLIT}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Output File: ${OUTPUT_FILE}"
echo "=============================================="

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT}" ]; then
    echo "Error: Checkpoint not found at ${CHECKPOINT}"
    exit 1
fi

# Check if data directory exists
if [ ! -d "${DATA_ROOT}" ]; then
    echo "Error: Data directory not found at ${DATA_ROOT}"
    exit 1
fi

# Run evaluation
python eval.py \
    --checkpoint "${CHECKPOINT}" \
    --data_root "${DATA_ROOT}" \
    --output_file "${OUTPUT_FILE}" \
    --batch_size ${BATCH_SIZE} \
    --split "${SPLIT}" \
    --clip_model "ViT-L/14" \
    --num_query_tokens 32 \
    --qformer_hidden_size 768 \
    --qformer_num_layers 12 \
    --max_txt_len 32 \
    --num_workers 4 \
    ${SKIP_RETRIEVAL}

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "Results saved to: ${OUTPUT_FILE}"
echo "=============================================="
