# Local LLM Toolkit

This repository hosts a number of scripts and CLI tools to:
- Collect, benchmark, and invoke local LLM models using Llama.cpp
- Indexing for Context (IfC) within a locally hosted codebase using ChromaDB
- Provide a user-specified number of relevant files as context with prompts
- Output results in a parsable manner
- (TBD) Provide scripts/tools to perform Low-Rank Adaptation (LoRA) fine-tuning of the indexed codebase on collected models
- (TBD) Improve quality of results by applying IfC and LoRA simultaneously

## Features
- **Indexing for Context (IFC)**: By running the Python indexing module, a ChromaDB index is produced which is queried by the model invoker script to pull relevant context for the prompt.
- **CLI Interface**: The CLI interface is provided by interacting with a bash script which accepts arguments and a prompt. Python modules are invoked to perform index queries and invoke the compiled Llama.cpp environment.
- **Flexible Model Selection**: Different GGUF models may be selected via CLI arguments. These are applied separately from the embedding model used for querying the index.
- **Scale-able Arguments**: When invoking Llama.cpp, several supported arguments are retained, such as LLM model, GPU thread count, and context size. Additional arguments to control index queries are also supported in the CLI layer.
- **Tested on AMD Hardware**: Llama.cpp was compiled on a moderate performance desktop with an AMD 7700X CPU, 7900GRE 16GB VRAM GPU, and 64GB RAM. Subjective: AMD hardware appears to involve a more restrictive installation, therefore this process may be easier or more compatible if using nVidia hardware.

## Background
...TODO

## Installation & Setup
...TODO

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
1. ...TODO
2. ...TODO Adjusting arguments
3. ...TODO "Unable to load model"

Example invocation:
```
/cli.sh -p "You are trained on the local codebase and have access to its index which includes a README file with examples. Create a structure from these examples which illustrates 3 features of the codebase." --topk 5 --gpulayers 24 --ctxsize 4096 --debug
```

## Notes on Specific Hardware
...TODO

## Output
...TODO

## Results
...TODO

## Conclusion
...TODO

## Future Enhancements
- ...TODO

## References
1. ...TODO
