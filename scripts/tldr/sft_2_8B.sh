#!/bin/bash
#SBATCH --job-name=sft_2_8b
#SBATCH --output=sft_2_8b.out
#SBATCH --partition=candle_9
#SBATCH --qos=candle_9_qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=2000G
#SBATCH --gres=gpu:H100:8
#SBATCH --time=2-00:00:00
srun --jobid $SLURM_JOB_ID bash -c 'source ~/.bashrc && conda activate rebel_2\
&& accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml \
--gpu_ids "0,1,2,3,4,5,6,7" \
--main_process_port 29084 \
--num_processes 8 \
src/tldr/sft.py \
--base_model EleutherAI/pythia-2.8b-deduped \
--output_dir /data/user_data/gswamy/models/models/sft_tldr_pythia_2.8b \
'