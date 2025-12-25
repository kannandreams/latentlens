# Vector Search Concepts

## 1. Scoring: Similarity vs. Distance

When you run a query, the system calculates how "close" your query is to the documents in the database.
There are two main ways this is represented:

### A. Cosine Similarity (The App's Metric)
- **Range**: -1.0 to 1.0
- **Meaning**: 1.0 means identical direction (most similar). 0 means orthogonal (unrelated). -1 means opposite.
- **Formula**:
$$
\\text{similarity} = \\cos(\\theta) = \\frac{\\mathbf{A} \\cdot \\mathbf{B}}{\\|\\mathbf{A}\\| \\|\\mathbf{B}\\|}
$$
- This app calculates this manually for the 3D visualization and ranking table to provide a consistent generic measure of similarity.

### B. DB Score (The Database's Metric)
- **Meaning**: Depends on the database configuration. 
- **Chroma Default**: Uses **Squared L2 Distance** (Euclidean).
- **Interpretation**: Lower is better (0 = identical).
- Unlike cosine similarity, distances can range from 0 to infinity (though usually small for normalized vectors).

---

## 2. Demo Embedder: Bag-of-Words

If you select the **Demo** connector, a "Toy Semantic" embedder is used. It is designed to be deterministic and lightweight for testing without needing an API key or heavy model.

**How it works:**
1. **Tokenize**: Splits text into words (e.g., "red shoes" -> ["red", "shoes"]).
2. **Hash & Project**: Each word is hashed to seed a random number generator, creating a consistent random 64-dimensional vector for that word.
3. **Sum**: Vectors of all words in the text are summed.
4. **Normalize**: The final vector is scaled to have a length of 1.

**Result**: 
- "red shoes" and "shoes red" produce **identical** vectors (order doesn't matter).
- "red shoes" and "blue shoes" share the "shoes" component, so they will be somewhat similar.
- "apple" and "galaxy" share no words, so their vectors will be (likely) orthogonal.

---

## 3. Dimensionality Reduction: 64D to 3D

Vectors often have hundreds of dimensions (e.g., 384 for MiniLM, 1536 for OpenAI). To show them on a screen, they must be reduced to 3 dimensions. This app uses a pipeline of two techniques:

### PCA (Principal Component Analysis)
- **What it is**: A linear technique that rotates the data to align with the axes of maximum variance.
- **Role**: It simplifies the data first, preserving the "global structure" (big picture).

### UMAP (Uniform Manifold Approximation and Projection)
- **What it is**: A non-linear graph-based algorithm.
- **Role**: It tries to preserve the "local neighborhood". If two points are close in high-dimensional space, UMAP tries very hard to keep them close in 3D.
- **Why**: It is excellent for showing clusters and groupings that linear PCA might miss.


---

## 4. Experimenting with Datasets

Preset "Challenge Datasets" are included in the **Manage Collection** tab to help demonstrate the difference between embedders.

### A. Synonyms (Semantic vs Literal)
- **Dataset**: Sentences like "The generated visuals were stunning" and "The created graphics were beautiful."
- **Demo (Bag-of-Words)**: Will likely see these as unrelated because they share almost no words ("The", "were").
- **MiniLM**: Will rank them together because it understands "visuals" ≈ "graphics" and "stunning" ≈ "beautiful".

### B. Negation (Sentiment)
- **Dataset**: "I love this" vs "I do not love this".
- **Demo**: Finds them very similar because they share 4 out of 5 words.
- **MiniLM**: Can discern the conflict in meaning, pushing them further apart (though they are still topically related).

### C. Polysemy (Context)
- **Dataset**: "Crane" (bird) vs "Crane" (machine).
- **Demo**: Confuses them entirely; a "crane" is a "crane".
- **MiniLM**: Uses the surrounding context ("lifted heavy beam" vs "waded in water") to disambiguate the vector.

---

## Further Reading

For a deeper dive into vector search and similarity, check out these resources:

- [Vector Similarity Techniques and Scoring (Elastic)](https://www.elastic.co/search-labs/blog/vector-similarity-techniques-and-scoring)
- [How Vector Similarity Search Works (Labelbox)](https://labelbox.com/blog/how-vector-similarity-search-works/)
- [Vector Similarity Explained (Pinecone)](https://www.pinecone.io/learn/vector-similarity/)

---
 
 ## 5. Troubleshooting & Warnings

 ### "Query appears isolated"
 This warning appears when the **maximum cosine similarity** between your query and the best result is very low (less than 0.15). 
 
 **What it means:**
 - **Demo Embedder**: Your query shares **zero words** with the documents. The matching score is essentially random noise (e.g., 0.04).
 - **Semantic Embedders (MiniLM/OpenAI)**: Your query is topically unrelated to the dataset (e.g., searching for "cooking recipes" in a "legal contracts" database).
 
 **How to fix:**
 - **Demo**: Use words that actually exist in the text.
 - **Semantic**: Ensure your query is relevant to the collection's domain.

 ---

## Glossary
- **Connector**: Backend that holds your vectors (Chroma is the standard local DB used here).
- **Embedder**: Model used to turn text into vectors (MiniLM local, OpenAI remote).
- **Top K results**: Number of nearest neighbors returned from the vector DB.
- **Background samples**: Random vectors for context, shown as gray points.
- **Void warning**: Indicator that the query sits far from retrieved results (possible mismatch).
