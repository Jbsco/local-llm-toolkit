#!/bin/bash
DIR=$1
# Config
# MODELS=("models/bench/sakura-solar-instruct.Q4_K_M.gguf")
#   "models/Q5/codellama-7b-instruct.Q4_K_M.gguf"
#   "models/Q5/llama-2-13b-chat.Q5_K_M.gguf"
#   "models/Q5/codellama-13b.Q5_K_M.gguf"
#   "models/Q5/mistral-7b-instruct-v0.1.Q5_K_M.gguf"
# )
MODELS=(models/"$DIR"/*.gguf)
GPU_LAYERS_VALUES=(24 36 48 60 72 84 96)  # adjust as needed based on VRAM headroom
OUTPUT_CSV="benchmark_results_24:12:96_${DIR}_models_$(date +%H:%M:%S).csv"
LLAMA_BIN="./llama.cpp/build/bin/llama-cli"
TOKENS=512
BATCH_SIZE=4096

# Check for rocm-smi command
if ! command -v /opt/rocm/bin/rocm-smi &> /dev/null; then
  echo "Warning: rocm-smi not found, VRAM usage won't be logged."
  log_vram() {
    echo "N/A"
  }
else
  log_vram() {
    /opt/rocm/bin/rocm-smi --showmeminfo vram | grep "Used" | awk '{print $(NF)/1e+06}'
  }
fi

# CSV Header
echo "model,gpu_layers,tokens_per_sec,total_ms,vram_before_MB,vram_after_MB" > $OUTPUT_CSV

for model in "${MODELS[@]}"; do
  echo "Benchmarking model: $model"
  for gpu_layers in "${GPU_LAYERS_VALUES[@]}"; do
    echo "  Running with --gpu-layers $gpu_layers..."

    vram_before=$(log_vram)

    # Run llama-cli, capture eval time per token and total time
    output=$($LLAMA_BIN -m "$model" --gpu-layers "$gpu_layers" --n-predict "$TOKENS" --ctx-size "$BATCH_SIZE" --color --jinja -st --prompt "user:\nbenchmark\nassistant:\n" 2>&1)

    vram_after=$(log_vram)

    # Parse output for eval_ms_per_token and total time
    eval_ms_per_token=$(echo "$output" | grep "llama_perf_context_print:        eval time" | awk '{print $(NF-3)}')
    total_ms=$(echo "$output" | grep "llama_perf_context_print:       total time" | awk '{print $(NF-4)}')

    # Fallback if no data found
    [[ -z "$eval_ms_per_token" ]] && eval_ms_per_token="N/A"
    [[ -z "$total_ms" ]] && total_ms="N/A"

    # Write to CSV
    echo "$(basename $model),$gpu_layers,$eval_ms_per_token,$total_ms,$vram_before,$vram_after" >> $OUTPUT_CSV

    echo "    tokens_per_sec $eval_ms_per_token, total_ms: $total_ms ms, VRAM before: $vram_before MB, VRAM after: $vram_after MB"
  done
done

echo "Benchmark complete. Results saved to $OUTPUT_CSV"
