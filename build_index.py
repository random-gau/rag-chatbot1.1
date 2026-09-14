"""
Build a FAISS index from PDFs in data/.

Usage:
    python build_index.py                          # defaults: chunk_size=500, overlap=50
    python build_index.py --chunk-size 1000 --chunk-overlap 200 --out faiss_index_1000_200

Kept as a CLI with explicit args (rather than hardcoded constants) on purpose:
step 3 needs to build multiple indexes with different chunking params to compare
retrieval quality, and re-editing a script each time is how "one main.py" turns
into five near-duplicate files. Ask if you want to see how that happened.
"""
import argparse
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdfs(pdf_dir: str):
    docs = []
    for fname in sorted(os.listdir(pdf_dir)):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir, fname)
            docs.extend(PyPDFLoader(path).load())
    return docs


def build_index(pdf_dir: str, out_dir: str, chunk_size: int, chunk_overlap: int):
    print(f"Loading PDFs from '{pdf_dir}'...")
    docs = load_pdfs(pdf_dir)
    print(f"Loaded {len(docs)} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap}).")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(out_dir)
    print(f"Saved FAISS index to '{out_dir}/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default="data")
    parser.add_argument("--out", default="faiss_index")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()

    build_index(args.pdf_dir, args.out, args.chunk_size, args.chunk_overlap)
