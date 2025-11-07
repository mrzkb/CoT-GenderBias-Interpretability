#!/bin/bash
#SBATCH --job-name=llama8b_bbq
#SBATCH --output=logs/llama8b_output_%j.txt
#SBATCH --error=logs/llama8b_error_%j.txt
#SBATCH --mem=32G # More GB due to attention monitoring
#SBATCH --gres=gpu:32gb:1
#SBATCH --time=02:00:00 # 1 hours, should be enough!

MODEL_PATH=$1
MODEL_NAME=$2
SAMPLE_SIZE=$3

if [ -z "$MODEL_PATH" ] || [ -z "$MODEL_NAME" ]; then
    echo "Usage: ./run.sh <model_path> <model_name>"
    echo "Example: ./run.sh /network/weights/llama.var/llama_3.1/Meta-Llama-3.1-8B llama3.1-8b"
    exit 1
fi

echo "Running pipeline for model: $MODEL_NAME"
echo "Model path: $MODEL_PATH"
if [ -n "$SAMPLE_SIZE" ]; then
    echo "Sample size: $SAMPLE_SIZE"
else
    echo "Sample size: Full dataset"
fi
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

module load python/3.10
source ~/venvs/bbq310/bin/activate

# Only pass sample_size if it's provided
if [ -n "$SAMPLE_SIZE" ]; then
    python 02.4_HERO.py \
        --model_path "$MODEL_PATH" \
        --model_name "$MODEL_NAME" \
        --sample_size "$SAMPLE_SIZE"
else
    python 02.4_HERO.py \
        --model_path "$MODEL_PATH" \
        --model_name "$MODEL_NAME"
fi

python 03.3_get_benchmark_performance.py \
    --model_name "$MODEL_NAME"