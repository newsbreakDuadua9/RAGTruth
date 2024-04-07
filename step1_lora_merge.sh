#!/bin/bash
echo "start merging lora weights!"

python models/merge_lora_weights.py \
--base_model meta-llama/Llama-2-13b-hf \
--peft_model models/model_file/ckpt/llama2_13b_lora_0217 \
--output_dir models/model_file/merged/llama2_13b_lora_0217_merge \
--safe_tensor True

echo "merge done!"