"""Streamlit UI entry point for Latent Lens (Vector Debugger)."""
from __future__ import annotations

import os
import uuid

import streamlit as st

from core.client_factory import build_client, generate_demo_context, get_embedder
from core.connectors import ChromaAdapter
from core.math_engine import detect_void_warning, reduce_query_context
from utils.visuals import build_scatter

st.set_page_config(page_title="Latent Lens — Vector Debugger", layout="wide")
st.title("🔍 Latent Lens")
st.caption("Vector Debugger")

if "show_help" not in st.session_state:
    st.session_state["show_help"] = False


def has_openai_key() -> bool:
    """Check for an OpenAI key in Streamlit secrets or environment variables."""
    return bool(st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def render_help_panel() -> None:
    st.info(
        """**How to use Latent Lens**
- Pick a connector and embedder in the sidebar. Demo generates synthetic data; Chroma reads/writes a real collection.
- Tune `Top K results` and `Background samples` to control retrieval depth and how much context appears in the 3D plot.
- Enter a query and click `Run Latent Lens` to embed the text and fetch nearest neighbors.
- In `Visualization`, optionally draw a distance ruler to a result and pick any point to inspect its metadata and scores.
- Using Chroma? Paste document text under "Chroma: store pasted text" to embed and save it into the active collection."""
    )


if st.button("How to use", help="Open a quick walkthrough without leaving the app."):
    st.session_state["show_help"] = not st.session_state["show_help"]

if st.session_state["show_help"]:
    render_help_panel()


with st.sidebar:
    st.header("Connector")
    connector = st.selectbox(
        "Database",
        ["Demo", "Chroma"],
        index=0,
        help="Pick where to fetch vectors. Demo creates synthetic data locally.",
    )
    top_k = st.slider(
        "Top K results",
        min_value=5,
        max_value=50,
        value=10,
        step=1,
        help="How many nearest neighbors to retrieve and plot.",
    )
    background_k = st.slider(
        "Background samples",
        min_value=100,
        max_value=1000,
        value=500,
        step=50,
        help="Extra random vectors for context so the 3D plot has shape.",
    )
    embedder_choice = st.selectbox(
        "Embedder",
        ["Demo", "MiniLM (local)", "OpenAI"],
        index=0,
        help="Embedding model used to encode the query (and any fetched records).",
    )
    run_button = st.button(
        "Run Latent Lens",
        help="Execute the retrieval + visualization with the current settings.",
    )
    openai_key_missing = embedder_choice == "OpenAI" and not has_openai_key()
    if openai_key_missing:
        st.warning(
            "OpenAI embedder selected but no `OPENAI_API_KEY` found. Set it in the environment or `st.secrets`."
        )

query = st.text_input(
    "Query text",
    value="red shoes",
    help="What you would search for. The app embeds this text and runs the retrieval.",
)

chroma_collection_name: str | None = None
if connector == "Chroma":
    chroma_collection_name = st.text_input(
        "Chroma Collection Name",
        value=st.session_state.get("chroma_collection_name", "default"),
        help="Name of the Chroma collection to read from and write to.",
        key="chroma_collection_name",
    )
    chroma_collection_name = (chroma_collection_name or "default").strip()

# Inline Chroma ingestion helper so users can paste raw text into the current collection.
if connector == "Chroma":
    st.subheader("Chroma: store pasted text")
    st.caption("Paste any document text to embed and store it in the selected Chroma collection.")
    with st.form("chroma_ingest_form"):
        pasted_text = st.text_area("Document text", placeholder="Copy/paste text to embed", height=160)
        doc_id_input = st.text_input(
            "Document ID (optional)",
            value=f"doc-{uuid.uuid4().hex[:8]}",
            help="Provide an ID or leave the default to avoid collisions.",
        )
        source_label = st.text_input(
            "Source label (optional)",
            value="clipboard",
            help="Optional tag to remember where this text came from.",
        )
        store_button = st.form_submit_button("Embed and store in Chroma")

    if store_button:
        if not pasted_text.strip():
            st.error("Please paste document text before storing it.")
        elif embedder_choice == "OpenAI" and openai_key_missing:
            st.error("Set `OPENAI_API_KEY` (env var or `st.secrets`) before using the OpenAI embedder.")
        else:
            try:
                embedder = get_embedder(embedder_choice)
                chroma_client, _ = build_client("Chroma", embedder, chroma_collection_name)
                if not isinstance(chroma_client, ChromaAdapter):
                    raise ValueError("Failed to initialize Chroma client.")
                doc_id = doc_id_input.strip() or f"doc-{uuid.uuid4().hex[:8]}"
                metadata = {"content": pasted_text.strip()}
                if source_label.strip():
                    metadata["source"] = source_label.strip()
                chroma_client.add_text(doc_id=doc_id, text=pasted_text.strip(), metadata=metadata)
                st.success(f"Stored '{doc_id}' in Chroma collection '{chroma_collection_name}'.")
            except Exception as exc:  # pragma: no cover - surfaces in UI
                st.error(f"Error storing document: {exc}")

# Keep latest run results so widget changes (like the ruler target) can update the plot
# without re-clicking the Run button.
if "viz_df" not in st.session_state:
    st.session_state["viz_df"] = None
if "viz_warning" not in st.session_state:
    st.session_state["viz_warning"] = None

if run_button:
    try:
        if connector == "Demo":
            ctx = generate_demo_context(query=query, top_k=top_k, background_k=background_k)
        else:
            if embedder_choice == "OpenAI" and openai_key_missing:
                raise ValueError("Set `OPENAI_API_KEY` in env or `st.secrets` to use the OpenAI embedder.")
            embedder = get_embedder(embedder_choice)
            client, name = build_client(connector, embedder, chroma_collection_name)
            with st.spinner(f"Querying {name}..."):
                ctx = client.retrieve_with_background(query=query, top_k=top_k, background_k=background_k)

        df = reduce_query_context(ctx)
        warning = detect_void_warning(df)

        st.session_state["viz_df"] = df
        st.session_state["viz_warning"] = warning
    except Exception as exc:  # pragma: no cover - surfaces in UI
        st.error(f"Error: {exc}")

df = st.session_state.get("viz_df")
warning = st.session_state.get("viz_warning")

if df is not None:
    st.subheader("Visualization")
    st.caption("3D projection of the query (red), retrieved results (blue), and background vectors (gray).")
    ruler_target = st.selectbox(
        "Distance ruler target (optional)",
        options=["None"] + df[df["label"] == "result"]["id"].tolist(),
        help="Draw an orange line from the query point to a chosen result to see relative distance.",
    )
    figure = build_scatter(df, ruler_target_id=None if ruler_target == "None" else ruler_target)

    st.plotly_chart(figure, use_container_width=True)

    selected_id = st.selectbox(
        "Inspect metadata for ID",
        df["id"].tolist(),
        help="Pick any plotted point to inspect its metadata and similarity details.",
    )

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
