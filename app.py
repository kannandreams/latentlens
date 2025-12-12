"""Streamlit UI entry point for the Vector Debugger."""
from __future__ import annotations

import os
import random
from typing import Any, Callable, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from core.connectors import (
    ChromaAdapter,
    PineconeAdapter,
    QdrantAdapter,
    QueryWithContext,
    VectorRecord,
    VectorDBClient,
)
from core.math_engine import detect_void_warning, reduce_query_context
from utils.visuals import build_scatter


st.set_page_config(page_title="Vector Debugger", layout="wide")
st.title("🔍 The Vector Debugger")


def _demo_embedder(text: str) -> list[float]:
    """Stable pseudo-embedding for demo purposes."""
    rng = random.Random(hash(text) % (2**32))
    return [rng.random() for _ in range(64)]


def _openai_embedder(model: str = "text-embedding-3-small") -> Callable[[str], list[float]]:
    from openai import OpenAI

    api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment or Streamlit secrets.")
    client = OpenAI(api_key=api_key)

    def _embed(text: str) -> list[float]:
        resp = client.embeddings.create(model=model, input=text)
        return resp.data[0].embedding

    return _embed


def _build_pinecone_client(embedder: Callable[[str], list[float]]) -> VectorDBClient:
    import pinecone

    api_key = st.secrets.get("PINECONE_API_KEY") or os.environ.get("PINECONE_API_KEY")
    env = st.secrets.get("PINECONE_ENV") or os.environ.get("PINECONE_ENVIRONMENT")
    index_name = st.session_state.get("pinecone_index") or st.text_input("Pinecone Index Name")
    if not api_key or not env or not index_name:
        raise ValueError("Missing Pinecone credentials or index name.")
    pinecone.init(api_key=api_key, environment=env)
    index = pinecone.Index(index_name)
    namespace = st.session_state.get("pinecone_namespace") or st.text_input("Namespace (optional)", "")
    return PineconeAdapter(index=index, namespace=namespace or None, embedder=embedder)


def _build_chroma_client(embedder: Callable[[str], list[float]]) -> VectorDBClient:
    import chromadb

    collection_name = st.text_input("Chroma Collection Name", value="default")
    client = chromadb.Client()
    collection = client.get_or_create_collection(collection_name)
    return ChromaAdapter(collection=collection, embedder=embedder)


def _build_qdrant_client(embedder: Callable[[str], list[float]]) -> VectorDBClient:
    from qdrant_client import QdrantClient

    url = st.text_input("Qdrant URL", value="http://localhost:6333")
    api_key = st.text_input("Qdrant API Key", value="", type="password")
    collection = st.text_input("Collection Name")
    if not collection:
        raise ValueError("Collection name is required for Qdrant.")
    client = QdrantClient(url=url, api_key=api_key or None)
    return QdrantAdapter(client=client, collection_name=collection, embedder=embedder)


def _generate_demo_context(query: str, top_k: int, background_k: int) -> QueryWithContext:
    """Create synthetic vectors for local demo."""
    dim = 64
    query_vector = np.random.normal(size=dim).tolist()
    results = []
    for i in range(top_k):
        vec = (np.array(query_vector) + np.random.normal(scale=0.2, size=dim)).tolist()
        results.append(
            VectorRecord(
                id=f"demo_result_{i}",
                vector=vec,
                metadata={"content": f"Synthetic result {i}", "source": "demo"},
                score=float(np.random.random()),
            )
        )

    background = []
    for i in range(background_k):
        background.append(
            VectorRecord(
                id=f"bg_{i}",
                vector=np.random.normal(size=dim).tolist(),
                metadata={"content": f"Background vector {i}"},
                score=None,
            )
        )
    return QueryWithContext(
        query=query,
        query_vector=query_vector,
        results=results,
        background=background,
    )


def build_client(connector: str, embedder: Callable[[str], list[float]]) -> Tuple[VectorDBClient, str]:
    if connector == "Pinecone":
        return _build_pinecone_client(embedder), "Pinecone"
    if connector == "Chroma":
        return _build_chroma_client(embedder), "Chroma"
    if connector == "Qdrant":
        return _build_qdrant_client(embedder), "Qdrant"
    raise ValueError(f"Unsupported connector: {connector}")


with st.sidebar:
    st.header("Connector")
    connector = st.selectbox("Database", ["Demo", "Pinecone", "Chroma", "Qdrant"], index=0)
    top_k = st.slider("Top K results", min_value=5, max_value=50, value=10, step=1)
    background_k = st.slider("Background samples", min_value=100, max_value=1000, value=500, step=50)
    embedder_choice = st.selectbox("Embedder", ["Demo", "OpenAI"], index=0)
    run_button = st.button("Run Debugger")

query = st.text_input("Query text", value="red shoes")

if run_button:
    try:
        if connector == "Demo":
            ctx = _generate_demo_context(query=query, top_k=top_k, background_k=background_k)
        else:
            embedder = _demo_embedder if embedder_choice == "Demo" else _openai_embedder()
            client, name = build_client(connector, embedder)
            with st.spinner(f"Querying {name}..."):
                ctx = client.retrieve_with_background(query=query, top_k=top_k, background_k=background_k)

        df = reduce_query_context(ctx)
        warning = detect_void_warning(df)

        st.subheader("Visualization")
        ruler_target = st.selectbox(
            "Distance ruler target (optional)", options=["None"] + df[df["label"] == "result"]["id"].tolist()
        )
        figure = build_scatter(df, ruler_target_id=None if ruler_target == "None" else ruler_target)

        # Try to enable click-to-inspect if the optional dependency is available.
        selected_id = None
        try:
            from streamlit_plotly_events import plotly_events

            click_data = plotly_events(figure, click_event=True, hover_event=False, override_height=600)
            if click_data:
                selected_id = click_data[0]["customdata"][0]
                st.success(f"Selected point: {selected_id}")
        except ModuleNotFoundError:
            st.info("Install `streamlit-plotly-events` for click-to-inspect. Using manual selector instead.")

        st.plotly_chart(figure, use_container_width=True)

        if not selected_id:
            selected_id = st.selectbox("Inspect metadata for ID", df["id"].tolist())

        if selected_id:
            point = df[df["id"] == selected_id].iloc[0]
            st.subheader("Metadata")
            st.json(point["metadata"])
            if point.get("cosine_sim_to_query") is not None:
                st.metric("Cosine similarity to query", f"{point['cosine_sim_to_query']:.3f}")
            if point.get("score") is not None:
                st.metric("DB score/distance", f"{point['score']:.3f}")

        if warning:
            st.warning(warning)
    except Exception as exc:  # pragma: no cover - surfaces in UI
        st.error(f"Error: {exc}")
