#!/usr/bin/env bash

MODEL="./models/Q4/codellama-7b-instruct.Q4_K_M.gguf"
PROMPT=""
TOP_K=0
GPU_LAYERS=32
CTX_SIZE=4096
TOKENS=-1
VECTOR_DB_DIR="./index/llama_index"   # where your Chroma/FAISS index lives
DEBUG=false
INTERACTIVE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model) MODEL="$2"; shift 2;;
        -p|--prompt) PROMPT="$2"; shift 2;;
        -k|--topk) TOP_K="$2"; shift 2;;
        --gpulayers) GPU_LAYERS="$2"; shift 2;;
        --ctxsize) CTX_SIZE="$2"; shift 2;;
        --tokens) TOKENS="$2"; shift 2;;
        --vectordb) VECTOR_DB_DIR="$2"; shift 2;;
        --debug) DEBUG=true; shift;;
        -i|--interactive) INTERACTIVE=true; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [[ -z "$PROMPT" ]]; then
  echo "No prompt, running in interactive mode with topk = 0."
  TOP_K=0
  INTERACTIVE=true
fi

echo "Using model: $MODEL"
echo "Prompt: $PROMPT"
echo "Top K: $TOP_K"
echo "GPU Layers: $GPU_LAYERS"
echo "Context Size: $CTX_SIZE"
echo "Predict Tokens: $TOKENS"
echo "Vector DB dir: $VECTOR_DB_DIR"
echo "Debug: $DEBUG"

# VRAM_BEFORE=$(rocm-smi --showmeminfo vram | grep "Used" | awk '{print $(NF)/1e+06}')

# Run Python query engine, get the final prompt with context injected + inference
python3 query_engine.py \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --topk "$TOP_K" \
  --gpulayers "$GPU_LAYERS" \
  --ctxsize "$CTX_SIZE" \
  --npredict "$TOKENS" \
  --vectordb "$VECTOR_DB_DIR" \
  $( [[ "$DEBUG" == true ]] && echo "--debug" ) \
  $( [[ "$INTERACTIVE" == true ]] && echo "--interactive" )
#
# VRAM_AFTER=$(rocm-smi --showmeminfo vram | grep "Used" | awk '{print $(NF)/1e+06}')
#
# echo "VRAM Before: ${VRAM_BEFORE} GiB"
# echo "VRAM After : ${VRAM_AFTER} GiB"
