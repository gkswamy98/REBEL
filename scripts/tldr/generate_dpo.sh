#!/bin/bash
#SBATCH --job-name=generate_dpo_1.4b
#SBATCH --output=generate_dpo_1.4b.out
#SBATCH --partition=candle_9
#SBATCH --qos=candle_9_qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=2-00:00:00
#SBATCH --requeue

srun --jobid $SLURM_JOB_ID bash -c 'source ~/.bashrc && conda activate rebel_2 \
&& python src/tldr/generate_vllm.py \
--model /data/user_data/gswamy/models/models/dpo_tldr_pythia_1.4b_1 \
--iter 1 \
--p 25 \
'