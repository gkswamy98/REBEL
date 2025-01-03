#!/bin/bash
#SBATCH --job-name=local_rm_1_4b
#SBATCH --output=local_rm_1_4b.out
#SBATCH --partition=candle_9
#SBATCH --qos=candle_9_qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=2000G
#SBATCH --gres=gpu:H100:8
#SBATCH --time=2-00:00:00

r1=$(( $RANDOM % 100000 ))
echo "Seed: $r1"

r2=$(( $RANDOM % 100000 ))
echo "Seed: $r2"

r3=$(( $RANDOM % 100000 ))
echo "Seed: $r3"

srun --jobid $SLURM_JOB_ID bash -c 'source ~/.bashrc && conda activate rebel_2 \
&& accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml \
--gpu_ids "0,1,2,3,4,5,6,7" \
--main_process_port 29084 \
--num_processes 8 \
src/tldr/local_rm.py \
--base_model  /data/user_data/gswamy/models/models/sft_tldr_pythia_1.4b \
--output_dir  /data/user_data/gswamy/models/models/local_rm_H_beta5_tldr_pythia_1.4b_1 \
--seed '${r1}' \
&& accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml \
--gpu_ids "0,1,2,3,4,5,6,7" \
--main_process_port 29084 \
--num_processes 8 \
src/tldr/local_rm.py \
--base_model  /data/user_data/gswamy/models/models/sft_tldr_pythia_1.4b \
--output_dir  /data/user_data/gswamy/models/models/local_rm_H_beta5_tldr_pythia_1.4b_2 \
--seed '${r2}' \
&& accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml \
--gpu_ids "0,1,2,3,4,5,6,7" \
--main_process_port 29084 \
--num_processes 8 \
src/tldr/local_rm.py \
--base_model  /data/user_data/gswamy/models/models/sft_tldr_pythia_1.4b \
--output_dir  /data/user_data/gswamy/models/models/local_rm_H_beta5_tldr_pythia_1.4b_3 \
--seed '${r3}' \
'