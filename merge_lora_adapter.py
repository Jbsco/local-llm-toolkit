from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL="models/Qwen2.5-Coder-7B-Instruct"  # Base model path
ADAPTER_MODEL="models/Qwen2.5-Coder-7b-Instruct-Adamant"  # Adapter path
MERGED_DIR="models/Qwen2.5-Adamant-Coder-7B-Instruct" # Output directory

# Load base and adapter
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto")
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)

# Merge weights
model = model.merge_and_unload()  # This permanently merges LoRA into base weights

# Save merged model
model.save_pretrained(MERGED_DIR)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(MERGED_DIR)
