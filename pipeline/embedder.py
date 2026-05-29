import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

_MODEL: SentenceTransformer | None = None
MODEL_NAME = "all-MiniLM-L6-v2"

def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        print(f"[embedder] Loading model '{MODEL_NAME}'...")
        _MODEL = SentenceTransformer(MODEL_NAME)
        print(f"[embedder] Model loaded.")
    return _MODEL

def embed(texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
    model = get_model()
    if isinstance(texts, str):
        texts = [texts]

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embeddings.astype(np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.squeeze(a)
    b = np.squeeze(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def centroid(embeddings: List[np.ndarray]) -> np.ndarray:
    stacked = np.vstack(embeddings)
    return stacked.mean(axis=0)
