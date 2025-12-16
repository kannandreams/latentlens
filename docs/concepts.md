# Vector Search Concepts

## 1. Scoring: Similarity vs. Distance

When you run a query, the system calculates how "close" your query is to the documents in the database.
There are two main ways we represent this:

### A. Cosine Similarity (The App's Metric)
- **Range**: -1.0 to 1.0
- **Meaning**: 1.0 means identical direction (most similar). 0 means orthogonal (unrelated). -1 means opposite.
- **Formula**:
$$
\\text{similarity} = \\cos(\\theta) = \\frac{\\mathbf{A} \\cdot \\mathbf{B}}{\\|\\mathbf{A}\\| \\|\\mathbf{B}\\|}
$$
- We calculate this manually effectively for the 3D visualization and ranking table to give you a consistent generic measure of similarity.

### B. DB Score (The Database's Metric)
- **Meaning**: Depends on the database configuration. 
- **Chroma Default**: Uses **Squared L2 Distance** (Euclidean).
- **Interpretation**: Lower is better (0 = identical).
- Unlike cosine similarity, distances can range from 0 to infinity (though usually small for normalized vectors).

---

## 2. Demo Embedder: Bag-of-Words

If you select the **Demo** connector, we use a "Toy Semantic" embedder. It is designed to be deterministic and lightweight for testing without needing an API key or heavy model.

**How it works:**
1. **Tokenize**: Splits text into words (e.g., "red shoes" -> ["red", "shoes"]).
2. **Hash & Project**: Each word is hashed to seed a random number generator, creating a consistent random 64-dimensional vector for that word.
3. **Sum**: We sum the vectors of all words in the text.
4. **Normalize**: The final vector is scaled to have a length of 1.

**Result**: 
- "red shoes" and "shoes red" produce **identical** vectors (order doesn't matter).
- "red shoes" and "blue shoes" share the "shoes" component, so they will be somewhat similar.
- "apple" and "galaxy" share no words, so their vectors will be (likely) orthogonal.

---

## 3. Dimensionality Reduction: 64D to 3D

Vectors often have hundreds of dimensions (e.g., 384 for MiniLM, 1536 for OpenAI). To show them on a screen, we must reduce them to 3 dimensions. We use a pipeline of two techniques:

### PCA (Principal Component Analysis)
- **What it is**: A linear technique that rotates the data to align with the axes of maximum variance.
- **Role**: It simplifies the data first, preserving the "global structure" (big picture).

### UMAP (Uniform Manifold Approximation and Projection)
- **What it is**: A non-linear graph-based algorithm.
- **Role**: It tries to preserve the "local neighborhood". If two points are close in high-dimensional space, UMAP tries very hard to keep them close in 3D.
- **Why**: It is excellent for showing clusters and groupings that linear PCA might miss.

---

## Glossary
- **Connector**: Backend that holds your vectors (Demo = synthetic, Chroma = local DB).
- **Embedder**: Model used to turn text into vectors (MiniLM local, OpenAI remote).
- **Top K results**: Number of nearest neighbors returned from the vector DB.
- **Background samples**: Random vectors for context, shown as gray points.
- **Void warning**: Indicator that the query sits far from retrieved results (possible mismatch).
