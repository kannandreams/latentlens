"""Helpers to build vector DB clients and demo data."""
from __future__ import annotations

import os
import random
from typing import Callable, Tuple

import numpy as np
import streamlit as st

from core.connectors import ChromaAdapter, QueryWithContext, VectorDBClient, VectorRecord


import hashlib

def demo_embedder(text: str) -> list[float]:
    """Stable pseudo-embedding using bag-of-words hash projection (MD5)."""
    # 1. Tokenize (simple whitespace)
    tokens = text.lower().split()
    if not tokens:
        # Return a consistent random vector for empty string or purely whitespace
        rng = random.Random(0)
        return [rng.random() for _ in range(64)]

    # 2. Sum vectors for each token
    dim = 64
    embedding = np.zeros(dim)
    for token in tokens:
        # Use MD5 for stable hashing across runs
        hash_digest = hashlib.md5(token.encode("utf-8")).digest()
        # Use first 4 bytes as seed
        seed_val = int.from_bytes(hash_digest[:4], "big")
        
        rng = random.Random(seed_val)
        # Generate random vector [-1, 1] for better orthogonality than [0, 1]
        token_vec = np.array([rng.uniform(-1, 1) for _ in range(dim)])
        embedding += token_vec

    # 3. Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm
    
    return embedding.tolist()


@st.cache_resource(show_spinner=False)
def _load_minilm(model_name: str) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def local_minilm_embedder(model: str = "sentence-transformers/paraphrase-MiniLM-L3-v2") -> Callable[[str], list[float]]:
    """Lightweight local embedder (256-dim MiniLM)."""
    encoder = _load_minilm(model)

    def _embed(text: str) -> list[float]:
        vector = encoder.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return vector.tolist()

    return _embed



def has_openai_key() -> bool:
    """Check if OPENAI_API_KEY is set in environment or secrets."""
    key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return bool(key)


def openai_embedder(model: str = "text-embedding-3-small") -> Callable[[str], list[float]]:
    from openai import OpenAI

    api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment or Streamlit secrets.")
    client = OpenAI(api_key=api_key)

    def _embed(text: str) -> list[float]:
        resp = client.embeddings.create(model=model, input=text)
        return resp.data[0].embedding

    return _embed


def get_embedder(choice: str) -> Callable[[str], list[float]]:
    if choice == "Demo":
        return demo_embedder
    if choice == "MiniLM (local)":
        return local_minilm_embedder()
    return openai_embedder()


def build_chroma_client(
    embedder: Callable[[str], list[float]], collection_name: str | None = None
) -> VectorDBClient:
    import chromadb

    collection_name = collection_name or st.session_state.get("chroma_collection_name")
    if not collection_name:
        collection_name = st.text_input(
            "Chroma Collection Name",
            value="default",
            help="Name of the Chroma collection to read vectors from.",
            key="chroma_collection_fallback",
        )
    client = chromadb.Client()
    collection = client.get_or_create_collection(collection_name)
    return ChromaAdapter(
        collection=collection,
        embedder=embedder,
    )


def build_client(
    connector: str, embedder: Callable[[str], list[float]], chroma_collection_name: str | None = None
) -> Tuple[VectorDBClient, str]:
    if connector == "Chroma":
        return build_chroma_client(embedder, collection_name=chroma_collection_name), "Chroma"
    raise ValueError(f"Unsupported connector: {connector}")




