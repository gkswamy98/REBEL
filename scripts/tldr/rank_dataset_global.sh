#!/bin/bash
#SBATCH --job-name=rank_dataset_global
#SBATCH --output=rank_dataset_global.out
#SBATCH --partition=candle_9
#SBATCH --qos=candle_9_qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=300G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=2-00:00:00
#SBATCH --requeue

model=$1
reward_model=$2

echo "Gen: $model"
echo "Reward Model: $reward_model"

r=$(( $RANDOM % 1000 + 29000 ))
echo "Random Port: $r"

srun --jobid $SLURM_JOB_ID bash -c 'source ~/.bashrc && conda activate rebel_2 \
&& accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml \
--gpu_ids "0" \
--main_process_port '${r}' \
--num_processes 1 \
src/tldr/process_vllm.py \
--model /data/user_data/gswamy/models/models/'${model}'  \
--reward_model /data/user_data/gswamy/models/models/'${reward_model}' \
--iter 1 \
--p 25 \
&& python src/tldr/merge_vllm.py \
--model /data/user_data/gswamy/models/models/'${model}'  \
--reward_model /data/user_data/gswamy/models/models/'${reward_model}' \
--iter 1 \
--p 25 \
'