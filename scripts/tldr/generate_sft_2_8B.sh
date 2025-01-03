#!/bin/bash
#SBATCH --job-name=generate_sft_2.8b
#SBATCH --output=generate_sft_2.8b.out
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
--model /data/user_data/gswamy/models/models/sft_tldr_pythia_2.8b \
--iter 1 \
--p 25 \
--prompt_aug True \
'