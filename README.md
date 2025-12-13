## Vector Debugger

Streamlit tool for inspecting embeddings across Pinecone, Chroma, and Qdrant with a 3D Plotly view.

![alt text](<vector debugger.gif>)

### Features
- Works with demo data or live Pinecone/Chroma/Qdrant connectors.
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
2) Enter the query text, adjust `Top K` and `Background samples`, then click **Run Debugger**.
3) Pan/zoom the 3D plot, select a distance ruler target if desired, and choose any ID to inspect metadata and similarity metrics. Warnings surface when the query appears isolated.
