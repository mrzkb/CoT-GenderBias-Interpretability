#!/bin/bash
#SBATCH --job-name=llama
#SBATCH --account=winter2025-comp598
#SBATCH --qos=comp579-1gpu-12h
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=llama_output.log
#SBATCH --error=llama_error.log
#SBATCH --nodelist=gpu-teach-03


module load miniconda/miniconda-winter2025
#pip install huggingface 
pip install scipy
pip install statsmodels
python run_llama.py

