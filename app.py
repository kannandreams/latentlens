"""Streamlit UI entry point for Latent Lens (Vector Debugger)."""
from __future__ import annotations

import os
import uuid

import streamlit as st

from core.client_factory import build_client, get_embedder, has_openai_key
from core.connectors import ChromaAdapter
from core.datasets import EXAMPLE_DATASETS
from core.math_engine import detect_void_warning, reduce_query_context
from utils.visuals import build_scatter
import base64

st.set_page_config(page_title="Latent Lens — Vector Debugger", layout="wide")
st.title("🔍 Latent Lens")
st.caption("Vector Debugger")

def has_openai_key() -> bool:
    """Check for an OpenAI key in Streamlit secrets or environment variables."""
    return bool(st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"))


APP_AUTHOR = "Kannan Kalidasan"
APP_WEBSITE = "https://kannandreams.github.io/"
APP_REPO = "https://github.com/kannandreams/latentlens"

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
    st.header("Configuration")
    # Hardcoded to Chroma as it is the only supported backend now.
    connector = "Chroma"

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
        type="primary",
        use_container_width=True
    )
    
    if "query_history" not in st.session_state:
        st.session_state["query_history"] = []

    if st.button("Clear Explorer History", help="Reset the 'ghost' path of previous searches in the Explorer tab.", use_container_width=True):
        st.session_state["query_history"] = []
        st.rerun()

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
        .cta-box {
            background-color: #f0f8ff;
            border: 1px solid #cce5ff;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 20px;
            font-size: 0.9rem;
        }
        .fundraising-section {
            background-color: #fff5f8;
            border: 1px solid #ffebeb;
            border-radius: 8px;
            padding: 12px;
            margin-top: 15px;
            text-align: center;
        }
        .fundraising-image {
            width: 100%;
            border-radius: 6px;
            margin-bottom: 10px;
        }
        .donate-btn {
            display: block;
            background-color: #92005a;
            color: white !important;
            text-align: center;
            padding: 10px 16px;
            text-decoration: none !important;
            border-radius: 5px;
            font-weight: bold;
            margin-top: 10px;
            transition: background-color 0.2s;
        }
        .donate-btn:hover {
            background-color: #7a004b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Base64 encode the fundraising image
    image_path = "assets/cwc_uk_fundraising.png"
    try:
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode()
        img_html = f'<img src="data:image/png;base64,{encoded_image}" class="fundraising-image" alt="Children with Cancer UK">'
    except FileNotFoundError:
        img_html = "" # Fallback if image missing

    st.markdown(
        f"""
        <div class="sidebar-footer">
            <div class="cta-box">
                If you like this project, give a star to the <a href="{APP_REPO}" target="_blank">repo</a>! ⭐️
            </div>
            <div style="margin-top: 0.5rem;">
                Built by <a href="{APP_WEBSITE}" target="_blank" rel="noopener noreferrer">{APP_AUTHOR}</a>
            </div>
            <div class="fundraising-section">
                {img_html}
                <p style="margin: 8px 0; line-height: 1.4; color: #333;">
                    Fundraising for <a href="https://www.childrenwithcancer.org.uk/" target="_blank">Children with Cancer UK</a> — thank you for your support 💛 
                </p>
                <a href="https://www.justgiving.com/page/kk-cwc-uk?utm_medium=FR&utm_source=CL" target="_blank" class="donate-btn">
                    Donate Now
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

chroma_collection_name = None
demo_tab, trajectory_tab, docs_tab, inspector_tab = st.tabs(["Explorer", "Query Trajectory", "Documentation", "Manage Collection"])

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


    # --- Load Example Dataset ---
    st.divider()
    st.subheader("Load Example Dataset")
    st.caption("Quickly populate the collection to test specific concepts.")
    
    col_ds, col_btn = st.columns([3, 1])
    with col_ds:
        selected_dataset = st.selectbox(
            "Choose a dataset", 
            options=list(EXAMPLE_DATASETS.keys()),
            key="dataset_selector"
        )
    with col_btn:
        st.write("")
        st.write("")
        load_ds_btn = st.button("Load Dataset", key="load_ds_btn")

    if load_ds_btn:
        try:
             embedder_ref = get_embedder(embedder_choice)
             c_client, _ = build_client("Chroma", embedder_ref, insp_collection)
             
             if isinstance(c_client, ChromaAdapter):
                 records = EXAMPLE_DATASETS[selected_dataset]
                 count = 0
                 for item in records:
                     doc_id = f"ex-{uuid.uuid4().hex[:6]}"
                     c_client.add_text(
                         doc_id=doc_id, 
                         text=item["content"], 
                         metadata=item.get("metadata")
                     )
                     count += 1
                 
                 st.success(f"Loaded {count} examples from '{selected_dataset}' into '{insp_collection}'.")
                 st.rerun()
             else:
                 st.warning("Loading examples only supported for Chroma.")

        except Exception as e:
            st.error(f"Error loading dataset: {e}")

    # --- Ingestion Form ---
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

with trajectory_tab:
    st.subheader("Query Trajectory")
    st.caption("How your query's meaning 'flies' through space as you add or change words.")

    with st.expander("📖 How to use this feature"):
        st.markdown("""
        **Query Trajectory** visualizes how adding specific words shifts the mathematical 'meaning' of your search.
        
        **Steps to try it:**
        1.  **Start Simple**: Type `Bank` and click **Add to Trajectory**. A red point appears.
        2.  **Add Context**: Type `Bank of the river` and click again.
        3.  **Watch the shift**: A dotted line will draw the 'flight' from the Finance cluster to the Nature cluster.
        4.  **Explore**: Try adding `River bank` or `Bank account` to see diverging paths!
        
        *Note: Past steps are shown as pink circles, the current step is a red diamond.*
        """)
    
    traj_query = st.text_input(
        "Add a step to the trajectory",
        placeholder="e.g. 'Bank', 'Bank of the river', 'Riverside bank'...",
        key="traj_input"
    )
    
    col_t1, col_t2 = st.columns([1, 4])
    with col_t1:
        add_traj = st.button("Add to Trajectory", type="primary", use_container_width=True)
    with col_t2:
        if st.checkbox("Show background context?", value=True, help="Include random background points for spatial reference."):
             traj_bg_k = background_k
        else:
             traj_bg_k = 0

    if add_traj and traj_query.strip():
        try:
            embedder = get_embedder(embedder_choice)
            client, _ = build_client(connector, embedder, st.session_state.get("insp_col_name", "default").strip())
            
            # 1. Embed current step
            vec = embedder(traj_query.strip())
            
            # 2. Add to dedicated traj history
            if "traj_history" not in st.session_state:
                st.session_state["traj_history"] = []
            
            st.session_state["traj_history"].append({"query": traj_query.strip(), "vector": vec})
            
            # 3. Retrieve context for the latest step (optional, but requested "The Wow Moment")
            ctx = client.retrieve_with_background(query=traj_query.strip(), top_k=top_k, background_k=traj_bg_k)
            
            # 4. Reduce WITH all traj steps as 'history'
            from core.connectors import VectorRecord, QueryWithContext
            traj_records = []
            # All but the last one are history
            for h in st.session_state["traj_history"][:-1]:
                 traj_records.append(
                     VectorRecord(id=f"traj-{h['query']}", vector=h['vector'], metadata={"query": h['query']})
                 )
            
            df_traj = reduce_query_context(ctx, history=traj_records)
            st.session_state["traj_df"] = df_traj
            st.rerun()

        except Exception as e:
            st.error(f"Trajectory error: {e}")

    # Display Trajectory
    if st.session_state.get("traj_history"):
        st.write("**Current Path:** " + " ➔ ".join([f"`{h['query']}`" for h in st.session_state["traj_history"]]))
        
        df_t = st.session_state.get("traj_df")
        if df_t is not None:
             fig_t = build_scatter(df_t)
             st.plotly_chart(fig_t, use_container_width=True)
        
        if st.button("Reset Trajectory", help="Clear the current multi-step path and start fresh."):
            st.session_state["traj_history"] = []
            st.session_state["traj_df"] = None
            st.rerun()
    else:
        st.info("Type a word above (e.g. 'Bank') and click 'Add to Trajectory' to start your path.")

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
            if embedder_choice == "OpenAI" and openai_key_missing:
                raise ValueError("Set `OPENAI_API_KEY` in env or `st.secrets` to use the OpenAI embedder.")
            embedder = get_embedder(embedder_choice)
            client, name = build_client(connector, embedder, chroma_collection_name)
            with st.spinner(f"Querying {name}..."):
                ctx = client.retrieve_with_background(query=query, top_k=top_k, background_k=background_k)

            from core.connectors import VectorRecord
            
            # Prepare history (last 5 previous queries)
            history_records = []
            for h in st.session_state["query_history"][-5:]:
                 history_records.append(
                     VectorRecord(id=f"history-{h['query']}", vector=h['vector'], metadata={"query": h['query']})
                 )

            df = reduce_query_context(ctx, history=history_records)
            
            # Save current for next run
            current_entry = {"query": query, "vector": ctx.query_vector}
            if not st.session_state["query_history"] or st.session_state["query_history"][-1]["query"] != query:
                st.session_state["query_history"].append(current_entry)
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

        if st.session_state.get("query_history"):
            with st.expander("Query History & Trajectory Log"):
                st.caption("Chronological path of your search 'thoughts'. Past queries are shown as diamonds/dots in the plot.")
                history_data = [
                    {"#": i+1, "Query": h["query"]} 
                    for i, h in enumerate(st.session_state["query_history"])
                ]
                st.table(history_data)

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
            
            st.divider()
            st.subheader("Explain Score")
            
            # 1. Token Overlap
            doc_content = point["metadata"].get("content", "")
            if doc_content:
                q_tokens = set(query.lower().split())
                d_tokens = set(str(doc_content).lower().split())
                overlap = q_tokens.intersection(d_tokens)
                
                if overlap:
                    st.success(f"Overlapping tokens: {', '.join(overlap)}")
                else:
                    st.warning("No overlapping tokens found.")
                    if point.get("cosine_sim_to_query", 0) > 0.3 and connector == "Chroma":
                         st.caption("High score without overlap? For 'Demo' embedder, this is likely a random hash collision. For semantic embedders (MiniLM/OpenAI), this indicates semantic similarity.")
            
            # 2. Word Similarity Heatmap
            if doc_content:
                st.caption("Word Similarity Heatmap: Pairwise cosine similarity.")
                q_words = [w for w in query.split() if w.strip()]
                d_words = [w for w in str(doc_content).split() if w.strip()][:15]  # Limit to 15 words
                
                if q_words and d_words:
                    import numpy as np
                    import pandas as pd
                    try:
                        # Re-instantiate a fresh embedder to check raw word vectors
                        # Note: This technically re-downloads/caches models if not already in memory, 
                        # but st.cache_resource handles it. 
                        # Ideally, we'd pass the embedder_ref, but it's local in scope. 
                        # We'll re-fetch it here safely.
                        raw_embedder = get_embedder(embedder_choice)
                        
                        heatmap_data = []
                        for q_w in q_words:
                            row_scores = []
                            v_q = np.array(raw_embedder(q_w))
                            for d_w in d_words:
                                v_d = np.array(raw_embedder(d_w))
                                # Cosine sim
                                score = np.dot(v_q, v_d) / (np.linalg.norm(v_q) * np.linalg.norm(v_d) + 1e-9)
                                row_scores.append(score)
                            heatmap_data.append(row_scores)
                        
                        df_heatmap = pd.DataFrame(heatmap_data, index=q_words, columns=d_words)
                        st.dataframe(df_heatmap.style.background_gradient(cmap="Reds", axis=None), use_container_width=True)
                        
                    except Exception as e:
                        st.warning(f"Could not generate heatmap: {e}")

            # 3. Vector Barcode
            query_point = df[df["label"] == "query"].iloc[0]
            if "vector" in point and point["vector"] is not None and "vector" in query_point:
                import pandas as pd
                st.caption("Vector Barcode: Visual comparison of the raw dimensions.")
                
                # Create a DataFrame for the bar chart
                vec_df = pd.DataFrame({
                    "Query": query_point["vector"],
                    "Result": point["vector"]
                })
                st.bar_chart(vec_df, height=200)
                
                with st.expander("How to read this chart?"):
                    st.markdown("""
                    **Visual comparison of raw dimensions**
                    *   **X-axis**: Dimension index (e.g., 0–63).
                    *   **Y-axis**: Numeric value.
                    *   **Dark Blue**: Query | **Light Blue**: Result.
                    
                    **Interpretation**:
                    *   **Good match**: Bars go up/down together (alignment).
                    *   **Weak match**: One positive, one negative (cancellation).
                    
                    *Note: Individual dimensions are abstract and do not map to specific words like "shoes".*
                    """)

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

