## Vector Debugger

Streamlit app for inspecting embeddings with a 3D Plotly view. Ships with a synthetic demo mode plus a Chroma connector and inline ingestion helper.

![Vector Debugger](<vector debugger.gif>)

### What it does
- Demo mode for offline experimentation; Chroma connector to inspect a live collection.
- Deterministic demo embedder, a local MiniLM (`sentence-transformers/paraphrase-MiniLM-L3-v2`), or OpenAI (`text-embedding-3-small`) for queries and Chroma writes.
- PCA → UMAP reduction into a 3D scatter: query (red diamond), results (blue), background (gray).
- Optional distance ruler from the query to any result, plus metadata, DB score, and cosine similarity readouts.
- Flags when the query looks isolated relative to retrieved neighbors.

### Requirements
- Python 3.9+ and `pip` (or `uv`; `uv pip install -r requirements.txt` also works).
- For OpenAI embeddings: set `OPENAI_API_KEY` (environment variable or `st.secrets`).
- For the local MiniLM embedder: first run will download the model via `sentence-transformers`.
- For Chroma: a collection name (defaults to `default`). The default client uses local storage; point it at your own Chroma instance if desired.

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run
```bash
streamlit run app.py
```

### How to use
1) Open the app and choose a connector in the sidebar: `Demo` (offline synthetic vectors) or `Chroma` (live collection). Pick an embedder (`Demo`, `MiniLM (local)`, or `OpenAI`).
2) For Chroma, set the collection name. Use **Chroma: store pasted text** to embed and add snippets directly into that collection (optional IDs and source tags supported).
3) Enter your query text, adjust `Top K results` and `Background samples`, then click **Run Debugger**.
4) Explore the 3D plot: pan/zoom, pick a `Distance ruler target` to see an orange line from the query to a chosen result, and select any ID to inspect metadata, cosine similarity to the query, and the DB score/distance.
5) If a warning appears, it signals the query is visually isolated—use it as a hint to tweak data or parameters.

### Notes
- The distance ruler measures straight-line distance in the reduced 3D projection, not the raw vector-space metric; rely on cosine similarity and DB scores for actual closeness.
- If a connector cannot return enough background vectors, the app pads with synthetic points so the 3D plot still has structure; those synthetic IDs are prefixed with `bg-synth-`.
- Only Demo and Chroma are wired into the UI today; Pinecone and Qdrant adapters live in `core/connectors.py` but are not exposed in `app.py`.
