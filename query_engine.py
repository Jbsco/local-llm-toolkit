import typer
import subprocess
from pathlib import Path
from typing import List

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage, SimpleDirectoryReader, Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings as ChromaSettings

app = typer.Typer()

def load_vector_index(vectordb_llama: str, vectordb_chroma: str) -> VectorStoreIndex:
    # Initialize local embedding model wrapper for llama_index
    # hf_embed = HuggingFaceEmbeddings(model_name="models/embedding/instructor-base")
    # hf_embed = HuggingFaceEmbeddings(model_name="models/embedding/gte-large")
    hf_embed = HuggingFaceEmbeddings(model_name="models/embedding/bge-large-en-v1.5")
    embed_model = LangchainEmbedding(hf_embed)

    # Create local persistent Chroma client
    chroma_client = chromadb.PersistentClient(path=vectordb_chroma)

    # Get or create collection
    chroma_collection = chroma_client.get_or_create_collection("codebase")

    # print("Number of vectors in collection:", chroma_collection.count())

    # Pass explicit collection to vector store to avoid HTTP fallback
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create Settings with embed_model
    Settings.embed_model = embed_model

    # Create storage context from vector store
    storage_context = StorageContext.from_defaults(
        persist_dir=vectordb_llama,
        vector_store=vector_store,
    )

    # Load index from storage context
    index = load_index_from_storage(storage_context)
    return index, vector_store

def retrieve_docs_from_chroma(vector_store, query, topk=3):
    # results = vector_store._collection.query(
    #     query_texts=[query],
    #     n_results=topk,
    # )

    # Compute embedding for query text
    # Initialize local embedding model wrapper for llama_index
    # hf_embed = HuggingFaceEmbeddings(model_name="models/embedding/instructor-base")
    # hf_embed = HuggingFaceEmbeddings(model_name="models/embedding/gte-large")
    hf_embed = HuggingFaceEmbeddings(model_name="models/embedding/bge-large-en-v1.5")
    embed_model = LangchainEmbedding(hf_embed)

    # Apply query embedding with BAAI recommended instruction:
    formatted_query = f"Represent this sentence for searching relevant passages: {query}"
    embedding = hf_embed.embed_documents([formatted_query])[0]

    results = vector_store._collection.query(
        query_embeddings=[embedding],
        n_results=topk,
        include=["documents", "metadatas", "distances"]
    )

    # results['documents'] is List[List[str]] (one list per query)
    docs_nested = results.get('documents', [])
    metas_nested = results.get("metadatas", [])
    if not docs_nested or not docs_nested[0]:
        return []

    docs = docs_nested[0]
    metas = metas_nested[0]

    # Return a list of tuples: (filename, document)
    return [(meta.get("source_file", "unknown"), doc) for meta, doc in zip(metas, docs)]

def construct_prompt_with_context(query: str, docs: list[tuple[str, str]]) -> str:
    if not docs:
        context_text = ""
    else:
        context_text = "\n\n".join(
            f"Filename: {filename}\nContext:\n{doc}" for filename, doc in docs
        )
    return f"<|context|>\n{context_text}\n<|user|>\n{query}\n<|assistant|>\n"

@app.command()
def main(
    model: str = typer.Option(..., help="GGUF model path"),
    prompt: str = typer.Option(..., help="User prompt"),
    topk: int = typer.Option(0, help="Top K similar docs"),
    gpulayers: int = typer.Option(40, help="Number of GPU layers for llama.cpp"),
    ctxsize: int = typer.Option(4096, help="Context size"),
    npredict: int = typer.Option(512, help="Tokens to predict"),
    vectordb: str = typer.Option("index/llama_index", help="Vector DB directory"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode."),
    interactive: bool = typer.Option(False, "--interactive", help="Enable interactive mode.")
):
    # Load vector index
    index, vector_store = load_vector_index(vectordb, "index/chroma")

    if topk > 0:
        # Retrieve docs from vector store
        docs = retrieve_docs_from_chroma(index.vector_store, prompt, topk)
        full_prompt = construct_prompt_with_context(prompt, docs)
    else:
        # Just use plain prompt, no context injection
        full_prompt = f"User:\n{prompt}\nAssistant:"

    # Call llama.cpp with the composed prompt
    llama_cmd = [
        "./llama.cpp/build/bin/llama-cli",
        "-m", model,
        "--gpu-layers", str(gpulayers),
        "--ctx-size", str(ctxsize),
        "--n-predict", str(npredict),
        "--color",
        # "--jinja",
        "-st",
        "--no-display-prompt",
    ] + (["-i"] if interactive else ["--prompt", full_prompt])

    if debug:
        print("[DEBUG] Running llama.cpp inference (live output):")
        print("[DEBUG] Full prompt:")
        print(full_prompt)

        process = subprocess.Popen(
            llama_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'  # safe decoding of output
        )
        full_output = ""
        for line in process.stdout:
            print(line, end="")  # live echo
            full_output += line
        process.wait()

    else:
        # Quieter mode, just the prompt and response
        result = subprocess.run(
            llama_cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'  # safe decoding here too
        )
        print(result.stdout)

if __name__ == "__main__":
    app()
