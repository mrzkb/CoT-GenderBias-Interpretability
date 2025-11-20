#!/bin/bash
#SBATCH --job-name=bbq
#SBATCH --output=logs/bbq_output_%j.txt
#SBATCH --error=logs/bbq_error_%j.txt
#SBATCH --mem=32G
#SBATCH --gres=gpu:32gb:1
#SBATCH --time=04:00:00 

module load python/3.10
source ~/venvs/bbq310/bin/activate

# python 02.5_soapbox.py --source bbq --unknown --instruction_version IV2 --sample_size 200
python 02.5_soapbox.py --source bbq --cot --unknown --instruction_version IV2 --sample_size 200
# python 02.5_soapbox.py --source stereo --unknown
# python 02.5_soapbox.py --source stereo --cot --unknown

# python 02.5_soapbox_eval.py