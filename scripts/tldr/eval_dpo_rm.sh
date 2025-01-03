#!/bin/bash
#SBATCH --job-name=eval_dpo_1.4rm
#SBATCH --output=eval_dpo_rm_1.4.out
#SBATCH --partition=candle_9
#SBATCH --qos=candle_9_qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=2-00:00:00

model=$1
echo "DPO RM: $model"
r=$(( $RANDOM % 1000 + 29000 ))
echo "Random Port: $r"

srun --jobid $SLURM_JOB_ID bash -c 'source ~/.bashrc && conda activate rebel_2 \
&& accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml \
--gpu_ids "0" \
--main_process_port '${r}' \
--num_processes 1 \
src/tldr/eval_local_rm.py \
--local_reward_model /data/user_data/gswamy/models/models/'${model}' \
--beta 0.05 \
--use_ref \
'