GPU=${0}

sudo docker run -d \
--name llama2_13b_lora_0217_merge \
--gpus "\"device=${GPU}\"" \
-v $PWD:/data \
--shm-size 1g \
-p 8321:80 ghcr.io/huggingface/text-generation-inference:1.3.4 \
--model-id /models/model_file/llama2_13b_lora_0217_merge \
--dtype bfloat16 \
--max-total-tokens 4096 \
--sharded false \
--max-input-length 4095 \
--cuda-memory-fraction 1
