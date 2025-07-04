#!/bin/bash -x

# === Paths ===
LORA_MODEL_DIR="models/Qwen2.5-Adamant-Coder-7B-Instruct"  # Adapter path
GGUF_OUTPUT="models/Qwen2.5-Adamant-Coder-7B-Instruct_" # GGUF filepath
#
# Merge LoRA adapter with base model, convert merged model to GGUF
# Assumes llama.cpp is built and llama-model-converter.py is available
python3 llama.cpp/convert_hf_to_gguf.py \
   $LORA_MODEL_DIR \
   --outfile "${GGUF_OUTPUT}Q8.gguf" \
   --outtype q8_0

python3 llama.cpp/convert_hf_to_gguf.py \
   $LORA_MODEL_DIR \
   --outfile "${GGUF_OUTPUT}F16.gguf" \
   --outtype f16
