# Latent Lens 🔍

**Latent Lens** is a powerful visual debugger and educational tool for exploring vector embeddings. It helps you peek inside the "black box" of semantic search by projecting high-dimensional vectors into an interactive 3D map.

![Latent Lens](<vector debugger.gif>)

### Key Features

#### 1. Explorer (Vector Debugging)
- **3D Projection**: PCA → UMAP reduction into an interactive 3D scatter plot.
- **Explain Score**: Demystify why two texts are similar with:
    - **Token Overlap**: Breakdown of shared words.
    - **Vector Barcode**: Visual comparison of raw high-dimensional values.
    - **Similarity Heatmap**: Pairwise word-level correlation heatmap.
- **Distance Ruler**: Measure straight-line distances in the projected space.

#### 2. Query Trajectory (Visualizing Thought)
- **Semantic Flight**: Watch your query "fly" through space as you add or change words.
- **Path Tracing**: Connection lines show the evolution of meaning (e.g., from a "Finance" cluster to a "Nature" cluster).
- **Trajectory Log**: A step-by-step history of your conceptual journey.

#### 3. Manage Collection
- **Dataset Presets**: Load "Challenge Datasets" to test specific semantic edge cases (e.g., Word Collisions).
- **Live Ingestion**: Embed and store custom text directly into a local Chroma collection.
- **Reset & Sync**: Easily clear collections or sync with the underlying vector database.

#### 4. Educational Documentation
- Built-in concepts guide explaining vector math, dimensionality reduction, and troubleshooting (e.g., "Void Warnings" when a query is isolated).

---

### Technical Setup

- **Deterministic Logic**: Uses stable hashing for the "Demo" embedder, ensuring reproducible results.
- **Multiple Embedders**: Support for Demo (synthetic), local MiniLM (`sentence-transformers`), and OpenAI (`text-embedding-3-small`).
- **Flexible Storage**: Ships with an in-memory/local ChromaDB adapter.
- **Modern UI**: Clean, Vercel-inspired light theme.

### Installation

```bash
# Clone and enter the repo
git clone https://github.com/kannandreams/latentlens
cd latentlens

# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Requirements
- Python 3.9+
- `matplotlib` (for heatmap rendering)
- `OPENAI_API_KEY` (optional, for OpenAI embeddings)

---
Built by [Kannan Kalidasan](https://kannandreams.github.io/) | If you like this project, give it a star! ⭐

