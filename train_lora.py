import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, AutoConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from tqdm import tqdm
# from torch.cuda.amp import autocast

# === CONFIG ===
MODEL_NAME = "models/Qwen2.5-Coder-7B-Instruct"
OUTPUT_DIR = "models/Qwen2.5-Coder-7b-Instruct-Adamant"
BLOCK_SIZE = 512
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_STEPS = 1000
SAVE_EVERY = 200
LORA_RANK = 32
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# torch.cuda.empty_cache()

# === LOAD TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# === LOAD MODEL ===
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

with init_empty_weights():
    base_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

device_map = infer_auto_device_map(
    base_model,
    max_memory={"cpu": "32GiB"},
    no_split_module_classes=["QWenBlock"]
)

base_model = load_checkpoint_and_dispatch(
    base_model,
    MODEL_NAME,
    device_map=device_map,
    offload_folder="offload",
    offload_state_dict=True
)

# device_map = infer_auto_device_map(
#     base_model,
#     max_memory={0: "6GiB", "cpu": "48GiB"},
#     no_split_module_classes=["QWenBlock"]
# )
#
# base_model = load_checkpoint_and_dispatch(
#     base_model,
#     MODEL_NAME,
#     device_map=device_map,
#     offload_folder="offload",
#     offload_state_dict=True
# )

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
base_model.gradient_checkpointing_disable()
model.config.use_cache = True
model.train()

# === LOAD DATASET ===
dataset = load_dataset("json", data_files="codebase_dataset.jsonl", split="train")
dataset = dataset.map(lambda x: {"text": x["text"]})

def tokenize(example):
    tokenized = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=BLOCK_SIZE,
        return_attention_mask=True,
        return_tensors=None,  # Let Datasets keep native format
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dset = dataset.map(
    tokenize,
    batched=False,
    remove_columns=["text", "path"],
    num_proc=1
)

train_loader = DataLoader(
    tokenized_dset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=default_data_collator
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=100)

# def move_to_device(x, device):
#     if isinstance(x, torch.Tensor):
#         return x.to(device)
#     elif isinstance(x, list):
#         # convert list to tensor before moving to device
#         return torch.tensor(x).to(device)
#     else:
#         return x

# === TRAIN LOOP ===
global_step = 0
accum_loss = 0.0
optimizer.zero_grad()

for epoch in range(999):
    for step, batch in enumerate(tqdm(train_loader)):
        # batch = {k: v.to(torch.float16) if v.dtype.is_floating_point else v for k, v in batch.items()}
        # batch = {k: v.to(model.device) for k, v in batch.items()}

        # Move to CPU & float32
        batch = {
            k: v.to(dtype=torch.bfloat16) if isinstance(v, torch.Tensor) and v.dtype.is_floating_point else v
            for k, v in batch.items()
        }

        print({k: v.shape for k, v in batch.items()})

        # torch.cuda.empty_cache()
        # with autocast(device_type="cuda", dtype=torch.float16):
            # outputs = model(**batch)
        outputs = model(**batch)
        loss = outputs.loss / GRAD_ACCUM_STEPS
        loss.backward()
        accum_loss += loss.item()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            print(f"Step {global_step} - Loss: {accum_loss:.4f}")
            accum_loss = 0.0

            if global_step % SAVE_EVERY == 0:
                save_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

        if global_step >= NUM_TRAIN_STEPS:
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print("Training complete.")
            exit()
