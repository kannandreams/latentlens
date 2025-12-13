## Vector Debugger

Streamlit UI plus Python back-end utilities for inspecting embeddings across Pinecone, Chroma, and Qdrant.

### Layout
- `app.py`: Streamlit entrypoint.
- `core/connectors.py`: Generic `VectorDBClient` and adapters for Pinecone/Chroma/Qdrant.
- `core/math_engine.py`: PCA → UMAP reducer and outlier detection helper.
- `utils/visuals.py`: Plotly 3D scatter helpers.

### Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```

Use the **Demo** connector for offline testing. To use a real DB, select Pinecone/Chroma/Qdrant, provide credentials, and enter a query. The UI will show:
- Blue points: top-k search results
- Grey points: random background sample
- Red diamond: query embedding

Click a point (requires `streamlit-plotly-events`) or pick an ID to inspect metadata and similarity metrics. A warning will surface when the query looks like an outlier.
