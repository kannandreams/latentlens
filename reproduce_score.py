
import numpy as np
import random

def demo_embedder(text: str) -> list[float]:
    """Stable pseudo-embedding using bag-of-words hash projection."""
    tokens = text.lower().split()
    if not tokens:
        rng = random.Random(0)
        return [rng.random() for _ in range(64)]

    dim = 64
    embedding = np.zeros(dim)
    for token in tokens:
        rng = random.Random(hash(token) % (2**32))
        token_vec = np.array([rng.uniform(-1, 1) for _ in range(dim)])
        embedding += token_vec

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm
    
    return embedding.tolist()

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

q = "red shoes"
d = "Bill Gates is a famous philanthropist."

v_q = demo_embedder(q)
v_d = demo_embedder(d)

score = cosine_similarity(v_q, v_d)
print(f"Query: {q}")
print(f"Doc: {d}")
print(f"Score: {score:.4f}")

q_tokens = set(q.lower().split())
d_tokens = set(d.lower().split())
print(f"Overlap: {q_tokens.intersection(d_tokens)}")
