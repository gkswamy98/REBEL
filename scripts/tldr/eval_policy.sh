#!/bin/bash
#SBATCH --job-name=eval_policy
#SBATCH --output=eval_policy.out
#SBATCH --partition=candle_9
#SBATCH --qos=candle_9_qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=2-00:00:00
#SBATCH --requeue

model=$1
echo "Model: $model"

r=$(( $RANDOM % 1000 + 29000 ))
echo "Random Port: $r"

srun --jobid $SLURM_JOB_ID bash -c 'source ~/.bashrc && conda activate rebel_2 \
&& accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml \
--gpu_ids "0" \
--main_process_port '${r}' \
--num_processes 1 \
src/tldr/eval_policy_vllm.py \
--base_model /data/user_data/gswamy/models/models/'${model}' \
&& python src/tldr/winrate.py --file_name /data/user_data/gswamy/eval_policy/'${model}'/table.csv \
' || scontrol requeue $SLURM_JOB_ID