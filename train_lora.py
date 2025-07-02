import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import TaskType
from accelerate import Accelerator
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

accelerator = Accelerator()

# === CONFIG ===
MODEL_NAME = "models/Qwen2.5-Coder-14B-Instruct"
DATA_PATH = "codebase.txt"  # merged raw code text
OUTPUT_DIR = "models/Qwen2.5-Coder-14b-Instruct-Adamant"
LORA_RANK = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
BLOCK_SIZE = 512

# === LOAD TOKENIZER & MODEL ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
# base_model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     device_map="auto",
#     low_cpu_mem_usage=False,  # disable meta tensor lazy loading
#     torch_dtype=torch.float16,  # Or .bfloat16 if supported
#     trust_remote_code=True
# )
# base_model = prepare_model_for_kbit_training(base_model)

with init_empty_weights():
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

device_map = infer_auto_device_map(
    base_model,
    max_memory={0: "14GiB", "cpu": "48GiB"},  # adjust based on your VRAM/RAM
    no_split_module_classes=["QWenBlock"]     # or appropriate block class if known
)

base_model = load_checkpoint_and_dispatch(
    base_model,
    MODEL_NAME,
    device_map=device_map,
    offload_folder="offload",     # or any temp folder for CPU offload
    offload_state_dict=True
)


base_model.gradient_checkpointing_enable()

# Explicitly move to GPU
# base_model = base_model.to("cuda")

# === ADD LORA ===
# lora_config = LoraConfig(
#     r=LORA_RANK,
#     lora_alpha=LORA_ALPHA,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=LORA_DROPOUT,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
for name, module in base_model.named_modules():
    if any(proj in name for proj in ["proj"]):
        print(name)

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(base_model, lora_config)

# === LOAD DATASET ===
def load_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

dataset = load_dataset("json", data_files="codebase_dataset.jsonl", split="train")
dataset = dataset.map(lambda x: {"text": x["text"]})

# def tokenize_function(examples):
#     return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=BLOCK_SIZE)
def tokenize_function(example):
    tokenized = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=BLOCK_SIZE,
        return_tensors=None,
        return_special_tokens_mask=False
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# tokenized_dset = dataset.map(tokenize_function, batched=True)
tokenized_dset = dataset.map(
    tokenize_function,
    batched=False,  # important: per-line processing
    remove_columns=["text", "path"],  # clean up unused fields
    num_proc=4  # parallelism, optional
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === TRAINING ARGS ===
# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=2,
#     learning_rate=2e-4,
#     num_train_epochs=3,
#     save_total_limit=1,
#     logging_steps=10,
#     bf16=True,
#     save_strategy="epoch",
#     logging_dir=f"{OUTPUT_DIR}/logs",
#     report_to="none",
# )
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # max_memory={ "cpu": "32GB", "cuda": "12GB" },
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    save_total_limit=2,
    logging_steps=10,
    fp16=True,
    bf16=False,  # or bf16=True if supported
    gradient_checkpointing=True,
    warmup_steps=100,
    max_steps=1000,
    save_steps=200
    # evaluation_strategy="steps",
    # eval_steps=200
)

# === TRAIN ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
