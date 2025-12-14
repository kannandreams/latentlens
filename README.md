## Vector Debugger

Streamlit tool for inspecting embeddings across Pinecone, Chroma, and Qdrant with a 3D Plotly view.

![alt text](<vector debugger.gif>)

### Features
- Works with demo data or live Pinecone/Chroma/Qdrant connectors.
- For Chroma, paste document text directly in the app to embed and store it in the chosen collection.
- Choose a deterministic demo embedder or OpenAI (`text-embedding-3-small`) for queries.
- PCA → UMAP reduction into a 3D scatter showing query (red), results (blue), and background (gray) with an optional distance ruler.
- Inspect metadata, DB scores, and cosine similarity for any plotted point; heuristics flag isolated queries.

### Requirements
- Python 3.9+ and `pip` (or `uv` if you prefer `uv pip install -r requirements.txt`).
- OpenAI: set `OPENAI_API_KEY` (env var or `st.secrets`).
- Pinecone: `PINECONE_API_KEY` plus `PINECONE_ENVIRONMENT` (or `PINECONE_ENV` in `st.secrets`); index/namespace are provided in the UI.
- Qdrant: cluster URL and optional API key entered in the UI.
- Chroma: collection name (defaults to `default`).

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
1) Pick a connector in the sidebar (`Demo` for offline synthetic data) and select your embedder.
2) If you chose Chroma, set the collection name and optionally paste text into **Chroma: store pasted text** to embed/store snippets.
3) Enter the query text, adjust `Top K` and `Background samples`, then click **Run Debugger**.
4) Pan/zoom the 3D plot, select a distance ruler target if desired, and choose any ID to inspect metadata and similarity metrics. Warnings surface when the query appears isolated.



• - The “distance ruler target” option simply draws an orange line from the query point (red diamond) to the result you pick in the 3D plot (app.py, utils/visuals.py). It measures straight-line (Euclidean)
    distance in the reduced 3D space produced by PCA → UMAP, not the raw vector DB distance. Use it to compare relative closeness between results; shorter line ≈ more similar in this visualization.
  - “Inspect metadata for ID” lets you pick any plotted point and see its stored metadata plus metrics: cosine_sim_to_query (computed on the original high‑dim vectors against the query) and score (whatever the DB
    returned—could be similarity or distance depending on the backend, as shown in core/math_engine.py and the connector adapters).
  - When you type “red shoes” as the query, the app embeds that text, retrieves neighbors, reduces all vectors to 3D, and plots them: query in red, results in blue, background in gray. The ruler distance is only
    an approximate visual cue in that 3D projection; rely on the cosine similarity and DB score for the actual numeric closeness.
