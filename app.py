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

def has_openai_key() -> bool:
    """Check for an OpenAI key in Streamlit secrets or environment variables."""
    return bool(st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"))


APP_AUTHOR = "Kannan Kalidasan"
APP_WEBSITE = "https://kannandreams.github.io/"

def render_help_panel() -> None:
    st.info(
        """**How to use Latent Lens**
- Pick a connector and embedder in the sidebar. Demo generates synthetic data; Chroma reads/writes a real collection.
- Tune `Top K results` and `Background samples` to control retrieval depth and how much context appears in the 3D plot.
- Enter a query and click `Run Latent Lens` to embed the text and fetch nearest neighbors.
- In `Visualization`, optionally draw a distance ruler to a result and pick any point to inspect its metadata and scores.
- Using Chroma? Paste document text under "Chroma: store pasted text" to embed and save it into the active collection."""
    )


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

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] .sidebar-footer {
            margin-top: 1.5rem;
            padding-top: 0.75rem;
            border-top: 1px solid #e6e6e6;
            font-size: 0.85rem;
            color: #6c757d;
        }
        [data-testid="stSidebar"] .sidebar-footer a {
            color: #0d6efd;
            text-decoration: underline;
            font-weight: 600;
        }
        [data-testid="stSidebar"] .sidebar-footer a:hover {
            color: #0b5ed7;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sidebar-footer">
            <div style="font-weight: 700;">About</div>
            <div>Latent Lens is a work in progress. Share feedback or contribute on GitHub.</div>
            <div style="margin-top: 0.5rem;">
                Made by <a href="{APP_WEBSITE}" target="_blank" rel="noopener noreferrer">{APP_AUTHOR}</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

chroma_collection_name = None
demo_tab, docs_tab, inspector_tab = st.tabs(["Demo", "Documentation", "Manage Collection"])

with inspector_tab:
    st.subheader("Manage Collection")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
            insp_collection = st.text_input("Collection Name", value="default", key="insp_col_name")
    with col2:
            st.write("") 
            st.write("")
            load_btn = st.button("Load Records", key="insp_load_btn")
    with col3:
            st.write("")
            st.write("")
            reset_btn = st.button("Reset Collection", key="insp_reset_btn", type="primary")

    if reset_btn:
        try:
             embedder_ref = get_embedder(embedder_choice)
             c_client, _ = build_client("Chroma", embedder_ref, insp_collection)
             if isinstance(c_client, ChromaAdapter):
                 count = c_client.reset_collection()
                 st.success(f"Deleted {count} records from '{insp_collection}'.")
                 st.rerun()
             else:
                 st.warning("Reset only supported for Chroma.")
        except Exception as e:
             st.error(f"Error resetting collection: {e}")

    # --- Ingestion Form (Moved here) ---
    st.divider()
    st.subheader("Add Document")
    st.caption("Paste text to embed and store in the collection defined above.")

    def ingest_document(embedder_choice, collection_name):
        text = st.session_state.get("chroma_ingest_text", "")
        doc_id = st.session_state.get("chroma_ingest_doc_id", "")
        source = st.session_state.get("chroma_ingest_source", "")
        
        if not text.strip():
            st.session_state["ingest_message"] = ("error", "Please paste document text before storing it.")
            return

        if embedder_choice == "OpenAI" and not has_openai_key():
                st.session_state["ingest_message"] = ("error", "Set `OPENAI_API_KEY` before using the OpenAI embedder.")
                return

        try:
            embedder = get_embedder(embedder_choice)
            chroma_client, _ = build_client("Chroma", embedder, collection_name)
            if not isinstance(chroma_client, ChromaAdapter):
                raise ValueError("Failed to initialize Chroma client.")
            
            final_doc_id = doc_id.strip() or f"doc-{uuid.uuid4().hex[:8]}"
            metadata = {"content": text.strip()}
            if source.strip():
                metadata["source"] = source.strip()
            
            chroma_client.add_text(doc_id=final_doc_id, text=text.strip(), metadata=metadata)
            
            st.session_state["chroma_ingest_text"] = ""
            st.session_state["chroma_ingest_doc_id"] = f"doc-{uuid.uuid4().hex[:8]}"
            st.session_state["ingest_message"] = ("success", f"Stored '{final_doc_id}' in Chroma collection '{collection_name}'.")
        except Exception as exc:
            st.session_state["ingest_message"] = ("error", f"Error storing document: {exc}")

    with st.form("chroma_ingest_form"):
        st.text_area(
            "Document text", 
            placeholder="Copy/paste text to embed", 
            height=160,
            key="chroma_ingest_text"
        )
        col_id, col_src = st.columns(2)
        with col_id:
            st.text_input(
                "Document ID (optional)",
                value=f"doc-{uuid.uuid4().hex[:8]}",
                help="Provide an ID or leave the default to avoid collisions.",
                key="chroma_ingest_doc_id"
            )
        with col_src:
            st.text_input(
                "Source label (optional)",
                value="clipboard",
                key="chroma_ingest_source"
            )
        st.form_submit_button(
            "Embed and store",
            on_click=ingest_document,
            args=(embedder_choice, insp_collection)
        )

    if "ingest_message" in st.session_state:
        kind, msg = st.session_state["ingest_message"]
        if kind == "success":
            st.success(msg)
        else:
            st.error(msg)
        del st.session_state["ingest_message"]

    st.divider()

    if load_btn:
        try:
            embedder_ref = get_embedder(embedder_choice)
            c_client, _ = build_client("Chroma", embedder_ref, insp_collection)
            
            if isinstance(c_client, ChromaAdapter):
                records = c_client.list_records(limit=100)
                if not records:
                    st.info(f"No records found in collection '{insp_collection}'.")
                else:
                    st.success(f"Found {len(records)} records.")
                    data = []
                    for r in records:
                        row = {"ID": r.id}
                        if r.metadata:
                            row.update(r.metadata)
                        data.append(row)
                    
                    st.dataframe(data, use_container_width=True)
            else:
                st.warning("Inspector only supports Chroma collections currently.")

        except Exception as e:
            st.error(f"Error loading collection: {e}")

with demo_tab:
    query = st.text_input(
        "Query text",
        value="red shoes",
        help="What you would search for. The app embeds this text and runs the retrieval.",
    )

    if connector == "Chroma":
        # Use the collection name from the inspector tab for retrieval
        # If the inspector tab hasn't been interacted with, it defaults to "default"
        chroma_collection_name = st.session_state.get("insp_col_name", "default").strip()

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
        except Exception as exc:
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

        st.subheader("Ranking")
        # Filter for results only and sort by similarity
        results_df = df[df["label"] == "result"].copy()
        if "cosine_sim_to_query" in results_df.columns:
            results_df = results_df.sort_values("cosine_sim_to_query", ascending=False)
        
        display_df = results_df.copy()
        display_df["content_snippet"] = display_df["metadata"].apply(lambda x: str(x.get("content", ""))[:100] + "..." if x.get("content") else str(x))
        
        cols_to_show = ["id", "cosine_sim_to_query", "score", "content_snippet"]
        cols_to_show = [c for c in cols_to_show if c in display_df.columns]
        
        st.dataframe(
            display_df[cols_to_show].rename(columns={
                "id": "ID",
                "cosine_sim_to_query": "Similarity",
                "score": "Distance/Score", 
                "content_snippet": "Content"
            }),
            use_container_width=True
        )

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

    with docs_tab:
        st.subheader("Documentation & Concepts")
        
        with st.expander("How to use Latent Lens", expanded=True):
            render_help_panel()

        try:
            with open("docs/concepts.md", "r") as f:
                concepts_md = f.read()
            st.markdown(concepts_md)
        except FileNotFoundError:
            st.error("Documentation file `docs/concepts.md` not found.")

