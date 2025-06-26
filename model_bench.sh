models=("models/codellama-7b-instruct.Q4_K_M" "models/codellama-13b.Q5_K_M" "models/llama-2-13b-chat.Q5_K_M")

for model in "${models[@]}"; do
  echo "=== Model: $model ==="
  for layers in 12 16 20 24 28 32; do
    echo "-- layers=$layers --"
    ./llama.cpp/build/bin/llama-cli -m ${model}.gguf \
      --gpu-layers $layers --ctx-size 4096 --n-predict 512 \
      --mlock --prompt "Benchmark"
  done
done
