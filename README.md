# Local LLM Toolkit

This repository hosts a number of scripts and CLI tools to:
- Collect, benchmark, and invoke local LLM models using Llama.cpp
- Indexing for Context (IfC) within a locally hosted codebase using ChromaDB
- Provide a user-specified number of relevant files as context with prompts
- Output results in a parsable manner
- Provide scripts/tools to perform Low-Rank Adaptation (LoRA) fine-tuning of the indexed codebase on collected models
- Improve quality of results by applying IfC and LoRA simultaneously

## Features
- **Indexing for Context (IFC)**: By running the Python indexing module, a ChromaDB index is produced which is queried by the model invoker script to pull relevant context for the prompt.
- **CLI Interface**: The CLI interface is provided by interacting with a bash script which accepts arguments and a prompt. Python modules are invoked to perform index queries and invoke the compiled Llama.cpp environment.
- **Flexible Model Selection**: Different GGUF models may be selected via CLI arguments. These are applied separately from the embedding model used for querying the index.
- **Scale-able Arguments**: When invoking Llama.cpp, several supported arguments are retained, such as LLM model, GPU thread count, and context size. Additional arguments to control index queries are also supported in the CLI layer.
- **Tested on AMD Hardware**: Llama.cpp was compiled on a moderate performance Linux desktop with an AMD 7700X CPU, 7900GRE 16GB VRAM GPU, and 64GB RAM. Subjective: AMD hardware appears to involve a more restrictive installation, therefore this process may be easier or more compatible if using nVidia hardware.

## Background
...TODO

## Installation & Setup
...TODO

...TODO Llama.cpp may require Python 3.10, as some module versions requested by Llama.cpp may not be compatible with Python 3.13.
```
sudo pacman -S python310
```

...TODO Setup a Python virtual environment.
```
python3.10 -m venv .venv
source .venv/bin/activate
```
   
Clone the Llama.cpp repository:
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

...TODO Install ROCM and/or BLAS requirements

Build [Llama.cpp](https://github.com/ggml-org/llama.cpp). The Llama.cpp arguments used for building on the AMD `gfx1100` system described above are:
```
cmake -B build -S . -DGGML_HIP=ON -DCMAKE_C_COMPILER=/opt/rocm/bin/hipcc -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -DCMAKE_C_FLAGS="--offload-arch=gfx1100" -DCMAKE_CXX_FLAGS="--offload-arch=gfx1100" -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/hip" -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGPU_TARGET=gfx1100 -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release
```
Add GGUF models. Many are available from [Hugging Face](https://huggingface.co). Some suggestions are provided. Benmchmarking and results from using this toolkit with these models may be included later.

For light performance systems, a 7B Q4 or Q5 KM model may perform well:

[CodeLlama-7B-Instruct-GGUF](https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF)

[Mistral-7B-Instruct-v0.2-code-ft-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-code-ft-GGUF)

[Qwen2.5-Coder-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF)

Moderately capable systems may get better quality responses with more context using larger 10B-14B models.:

[Sakura-SOLAR-10.7B-Instruct-GGUF](https://huggingface.co/TheBloke/Sakura-SOLAR-Instruct-GGUF)

[CodeLlama-13B-Instruct-GGUF](https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF)

[Qwen2.5-Coder-14B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF)

Larger 30B+ models may also function and can handle longer, more complex tasks at the expense of system resources and time:

[Qwen2.5-32B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF)

[CodeLlama-34B-Instruct-GGUF](https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF)

## Usage
1. Running the CLI with a locally hosted model:
Download GGUF models in sizes and quants appropriate for the system. The `--model` argument accepts a path relative to wherever `cli.sh` is run from.
Place a copy of the codebase to index in the `codebase/` directory, and run the `index_codebase.py` script:
```
python index_codebase.py
```
This may take several minutes. After completion, the CLI script may be invoked:
```
./cli.sh -p "prompt" --gpulayers 40 --topk 3 --ctxsize 2056 --tokens 512 --model models/Q5/qwen2.5-coder-32b-instruct-q5_k_m.gguf --debug
```
Arguments, paths, and flags should be adjusted accordingly. `--debug` output includes the collected files for context, as well as the Llama.cpp initialization output.
3. ...TODO Adjusting arguments
4. ...TODO "Unable to load model"
5. LoRA training
Build the dataset:
```
python3 build_dataset_jsonl.py
```
Set the model and output in the `train_lora.py` script. Start training:
```
accelerate launch train_lora.py
```
Merge the base model and adapter, and create the `*.gguf` file(s):
```
python3 merge_lora_adapter.py
./merge_quantize_gguf.sh
```
The model may be used if these processes completed successfully.

Example invocation:
```
/cli.sh -p "You are trained on the local codebase and have access to its index which includes a README file with examples. Create a structure from these examples which illustrates 3 features of the codebase." --topk 5 --gpulayers 24 --ctxsize 4096 --debug
```

## Notes on Specific Hardware
Llama.cpp was built for a desktop Linux system using an AMD GPU. ROCM and HIP are enabled to leverage AMD `gfx1100` GPU hardware. Some flags are used for the makefile build, these are described in Installation & Setup. Nvidia hardware would benefit from a default build configuration. AMD GPUs can be capable here but are not the common case for AI development, Nvidia GPUs are recommended for any AI-purposed computer build.

LoRA training encounters compatibility restrictions as well. In testing, GPU utilization appeared possible even with only 16GiB VRAM, but all attempts failed due to lack of ROCM support at one stage or another. `bitsandbytes`, in particular, is a critical package used in quantization of models with Torch, but at this time ROCM compatibility is limited for the `gfx1100` architecture. Offloading and batching support was also found to be limited with ROCM. CPU training is still possible, but time consuming and best left to run unattended or overnight. CPU training is also limited in that, without `bitsandbytes` support and related features, the size of the model must be small in order to fit within system RAM, initially the only `qwen2.5-coder-instruct` model able to be trained is the 7B size. Even with 64Gb RAM the 14B size will easily exceed available memory and fail during initialization of the training script.

## Output
```
./cli.sh -p "you are trained on the Adamant framework and have access to an index of its source code, as well as examples of its yaml schema and numerous records. generate a taml record and ada .ads spec for a quaternion and cuc (ccsds unsegmented code) timestamp" --gpulayers 40 --topk 4 --ctxsize 4096 --tokens 4096 --model models/Q5/qwen2.5-coder-32b-instruct-q5_k_m.gguf
```
> Certainly! Below, I provide a TAML (Text-based Ada Modeling Language) record definition for a Quaternion and a CUC (CCSDS Unsegmented Code) timestamp, followed by the corresponding Ada `.ads` (specification) file for these records.
>
> ### TAML Record Definition
>
> First, let's define the TAML records for the Quaternion and the CUC timestamp. TAML is a YAML-based schema used in Adamant to define the structure of the records.
>
> ```
> yaml
> - Quaternion:
>     type: record
>     fields:
>       - name: w
>         type: f32
>         description: Scalar component of the quaternion
>       - name: x
>         type: f32
>         description: First vector component of the quaternion
>       - name: y
>         type: f32
>         description: Second vector component of the quaternion
>       - name: z
>         type: f32
>         description: Third vector component of the quaternion
> 
> - CucTimestamp:
>     type: record
>     fields:
>       - name: coarser
>         type: u16
>         description: Coarser part of the timestamp (seconds)
>       - name: finer
>         type: u16
>         description: Finer part of the timestamp (sub-seconds)
> ```
> ### Ada `.ads` Specification
> 
> Now, let's generate the corresponding Ada `.ads` specification for these records. These records will be packed to ensure they match the expected binary layout.
> ```
> ada
> -- Quaternion.ads
> --------------------------------------------------------------------------------
> -- Quaternion Record Specification
> --------------------------------------------------------------------------------
> 
> package Quaternion is
> 
>    -- Define the Quaternion record with packed fields
>    type T is packed record
>       W : Float;    -- Scalar component of the quaternion
>       X : Float;    -- First vector component of the quaternion
>       Y : Float;    -- Second vector component of the quaternion
>       Z : Float;    -- Third vector component of the quaternion
>    end record;
> 
> end Quaternion;
> ```
> 
> ```
> ada
> -- CucTimestamp.ads
> --------------------------------------------------------------------------------
> -- CUC Timestamp Record Specification
> --------------------------------------------------------------------------------
> 
> package CucTimestamp is
> 
>    -- Define the CUC Timestamp record with packed fields
>    type T is packed record
>       Coarser : Unsigned_16;  -- Coarser part of the timestamp (seconds)
>       Finer   : Unsigned_16;  -- Finer part of the timestamp (sub-seconds)
>    end record;
> 
> end CucTimestamp;
> ```
> ### Explanation
> 
> - **Quaternion Record**: The `Quaternion` record contains four fields: `W`, `X`, `Y`, and `Z`, each of type `Float` (which corresponds to the `f32` type in TAML). The record is packed to ensure that it is represented in a contiguous block of memory without any padding.
> 
> - **CUC Timestamp Record**: The `CucTimestamp` record contains two fields: `Coarser` and `Finer`, each of type `Unsigned_16`. The `Unsigned_16` type is assumed to be a predefined type in Ada that represents an unsigned 16-bit integer. This record is also packed for the same reasons as the `Quaternion` record.
> 
> These specifications can be used within the Adamant framework to define and manipulate Quaternion and CUC timestamp data structures in a consistent and efficient manner.
## Results
...TODO

## Conclusion
...TODO

## Future Enhancements
- ...TODO LoRA fine-tuning

## References
1. ...TODO
