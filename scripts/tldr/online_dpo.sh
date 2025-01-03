#!/bin/bash
#SBATCH --job-name=online_dpo_1_4
#SBATCH --output=online_dpo_1_4b.out
#SBATCH --partition=candle_9
#SBATCH --qos=candle_9_qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=2000G
#SBATCH --gres=gpu:H100:8
#SBATCH --time=2-00:00:00

model=$1
reward_model=$2

echo "Base Model: $model"
echo "Reward Model: $reward_model"

r1=$(( $RANDOM % 100000 ))
echo "Seed: $r1"

srun --jobid $SLURM_JOB_ID bash -c 'source ~/.bashrc && conda activate rebel_2\
&& accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml \
--gpu_ids "0,1,2,3,4,5,6,7" \
--main_process_port 29084 \
--num_processes 8 \
src/tldr/dpo_fix.py \
--base_model  /data/user_data/gswamy/models/models/'${model}' \
--reward_model /data/user_data/gswamy/models/models/'${reward_model}' \
--output_dir  /data/user_data/gswamy/models/models/online_dpo_'${model}'_'${reward_model}' \
--seed '${r1}' \
--sft_aug
'