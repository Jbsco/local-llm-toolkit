import os
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

from llama_index.core import StorageContext, SimpleDirectoryReader, Settings, Document, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings as ChromaSettings

# Helper to recursively read Ada + Python source files
def read_code_files(root_dir: str) -> List[Tuple[Path, str]]:
    exts = ['.adb', '.ads', '.py', '.yaml', '.yml', '.md', '.tex', '.sh', '.dockerfile', '.gpr', '.cpp', '.h']
    file_texts = []

    for ext in exts:
        for path in Path(root_dir).rglob(f"*{ext}"):
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    file_texts.append((path, text))  # <-- return (path, content)
            except Exception as e:
                print(f"Could not read {path}: {e}")
    return file_texts

def main():
    code_root = "./codebase"       # your code directory
    chroma_index_dir = "./index/chroma"
    llama_index_dir = "./index/llama_index"

    # Read and wrap source files as Document objects
    file_texts = read_code_files(code_root)
    print(f"Read {len(file_texts)} source files")

    # Convert to LlamaIndex Documents with progress bar
    documents = []
    for path, text in tqdm(file_texts, desc="Wrapping files as Documents"):
        documents.append(Document(
            text=text,
            metadata={"source_file": str(path)}  # use path.name for filename only
        ))

    # Wrap the HF embedding with Langchain-compatible wrapper
    # hf_embed = HuggingFaceEmbeddings(model_name="models/embedding/instructor-base")
    # hf_embed = HuggingFaceEmbeddings(model_name="models/embedding/gte-large")
    hf_embed = HuggingFaceEmbeddings(model_name="models/embedding/bge-large-en-v1.5")
    embed_model = LangchainEmbedding(hf_embed)

    # Service context from embeddings
    Settings.embed_model = embed_model

    # Create Chroma DB client
    chroma_client = chromadb.PersistentClient(path=chroma_index_dir)

    # Explicit collection to avoid fallback to HttpClient
    chroma_collection = chroma_client.get_or_create_collection("codebase")

    # Now wrap into LlamaIndex vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create storage context with that vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Index with progress bar
    print(f"Indexing {len(documents)} documents...")
    index = VectorStoreIndex.from_documents(
        tqdm(documents, desc="Embedding + indexing"),
        vector_store=vector_store,
       storage_context=storage_context,
    )
    # print("Vectors in collection:", chroma_collection.count())

    # Save index
    index.storage_context.persist(persist_dir=llama_index_dir)
    print(f"Index saved to {llama_index_dir}")

if __name__ == "__main__":
    main()
