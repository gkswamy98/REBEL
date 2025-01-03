#!/bin/bash
#SBATCH --job-name=eval_rm
#SBATCH --output=eval_rm.out
#SBATCH --partition=candle_9
#SBATCH --qos=candle_9_qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=2-00:00:00

model=$1
echo "Global RM: $model"
# # --reward_model_path /data/user_data/gswamy/models/models/'${model}' 
r=$(( $RANDOM % 1000 + 29000 ))
echo "Random Port: $r"

srun --jobid $SLURM_JOB_ID bash -c 'source ~/.bashrc && conda activate rebel_2 \
&& accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml \
--gpu_ids "0" \
--main_process_port '${r}' \
--num_processes 1 \
src/tldr/eval_global_rm.py \
--reward_model_path /data/user_data/gswamy/models/models/'${model}' 
'