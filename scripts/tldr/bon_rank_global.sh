#!/bin/bash
#SBATCH --job-name=eval_bon
#SBATCH --output=eval_bon.out
#SBATCH --partition=candle_9
#SBATCH --qos=candle_9_qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=2-00:00:00

gen=$1
reward_model=$2

echo "Gen: $gen"
echo "Reward Model: $reward_model"

r=$(( $RANDOM % 1000 + 29000 ))
echo "Random Port: $r"

srun --jobid $SLURM_JOB_ID bash -c 'source ~/.bashrc && conda activate rebel_2 \
&& accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml \
--gpu_ids "0" \
--main_process_port '${r}' \
--num_processes 1 \
src/tldr/bon_rank.py \
--eval_df_path /data/user_data/gswamy/bon/'${gen}'/table.csv \
--reward_model /data/user_data/gswamy/models/models/'${reward_model}' \
&& python src/tldr/winrate.py --file_name /data/user_data/gswamy/eval_bon/'${gen}'/'${reward_model}'/table_1.csv  --n 1 \
&& python src/tldr/winrate.py --file_name /data/user_data/gswamy/eval_bon/'${gen}'/'${reward_model}'/table_2.csv  --n 2 \
&& python src/tldr/winrate.py --file_name /data/user_data/gswamy/eval_bon/'${gen}'/'${reward_model}'/table_5.csv  --n 5 \
&& python src/tldr/winrate.py --file_name /data/user_data/gswamy/eval_bon/'${gen}'/'${reward_model}'/table_10.csv --n 10 \
&& python src/tldr/winrate.py --file_name /data/user_data/gswamy/eval_bon/'${gen}'/'${reward_model}'/table_25.csv --n 25 \
'