#!/bin/bash
echo "start merging lora weights!"

CUDA_VISIBLE_DEVICES=1 \
python models/merge_lora_weights.py \
--base_model meta-llama/Llama-2-13b-hf \
--peft_model models/model_file/llama2_13b_lora_0217_ckpt \
--output_dir models/model_file/llama2_13b_lora_0217_ckpt_merge \
--safe_tensor True

echo "merge done!"

