model=$1
iter=$2
gpus=$3
echo "Model: $model"
echo "Iter: $iter"
echo "GPUs: $gpus"

for gpu in $(seq 0 "$((gpus - 1))"); do
    CUDA_VISIBLE_DEVICES="$((gpu * 1))" python src/tldr/generate_vllm.py --model "$model" --iter "$iter" --p "$gpu" && \
    accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --gpu_ids "$((gpu * 1))" --main_process_port "$((gpu + 29080))" --num_processes 1  src/tldr/process_vllm.py --iter "$iter" --p "$gpu" &
    # CUDA_VISIBLE_DEVICES="$gpu" python src/tldr/generate_vllm.py --model "$model" --iter "$iter" --p "$((gpu + gpus))" && \
    # accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --gpu_ids "$gpu" --main_process_port "$((gpu + 29080))" --num_processes 1  src/tldr/process_vllm.py --iter "$iter" --p "$((gpu + gpus))" &
done

wait

python src/tldr/merge_vllm.py --iter "$iter" --p "$((gpus * 1))"