"""
pipeline/retriever.py

Two-stage retrieval combining topic-level summaries and raw message chunks.

Stage 1: Topic summary search (high-level thematic context)
Stage 2: Raw chunk search (supporting verbatim evidence)
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import faiss
import numpy as np

from pipeline.loader import Message
from pipeline.embedder import embed

CHUNK_SIZE = 20
CHUNK_OVERLAP = 5
TOP_K_TOPICS = 3
TOP_K_CHUNKS = 5

class Retriever:
    def __init__(self):
        self.topic_index: Optional[faiss.IndexFlatIP] = None
        self.topic_summaries: List[dict] = []

        self.chunk_index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[dict] = []

        self._dim: Optional[int] = None

    def build_topic_index(self, topic_checkpoints: List[dict]) -> None:
        if not topic_checkpoints:
            print("[retriever] No topic checkpoints — skipping topic index build.")
            return

        embeddings = [np.array(tc["embedding"], dtype=np.float32) for tc in topic_checkpoints]
        self._dim = embeddings[0].shape[0]

        matrix = np.vstack(embeddings)
        self.topic_index = faiss.IndexFlatIP(self._dim)
        self.topic_index.add(matrix)
        self.topic_summaries = topic_checkpoints
        print(f"[retriever] Topic index built: {len(topic_checkpoints)} summaries, dim={self._dim}")

    def build_chunk_index(self, messages: List[Message]) -> None:
        chunks_text = []
        chunks_meta = []

        i = 0
        stride = CHUNK_SIZE - CHUNK_OVERLAP
        while i < len(messages):
            window = messages[i:i + CHUNK_SIZE]
            chunk_text = " ".join(m.text for m in window)
            chunks_text.append(chunk_text)
            chunks_meta.append({
                "start_msg": window[0].message_id,
                "end_msg": window[-1].message_id,
                "text": "\n".join(f"{m.speaker}: {m.text}" for m in window),
            })
            i += stride

        print(f"[retriever] Embedding {len(chunks_text)} raw chunks (batch mode)...")
        embeddings = embed(chunks_text, normalize=True)

        if self._dim is None:
            self._dim = embeddings.shape[1]

        self.chunk_index = faiss.IndexFlatIP(self._dim)
        self.chunk_index.add(embeddings)
        self.chunks = chunks_meta
        print(f"[retriever] Chunk index built: {len(chunks_meta)} chunks, dim={self._dim}")

    def query(self, query_text: str) -> Tuple[str, List[dict], List[dict]]:
        q_emb = embed(query_text, normalize=True)

        topic_hits: List[dict] = []
        chunk_hits: List[dict] = []

        if self.topic_index is not None and self.topic_index.ntotal > 0:
            k = min(TOP_K_TOPICS, self.topic_index.ntotal)
            scores, indices = self.topic_index.search(q_emb, k)
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    hit = dict(self.topic_summaries[idx])
                    hit["retrieval_score"] = float(score)
                    topic_hits.append(hit)

        if self.chunk_index is not None and self.chunk_index.ntotal > 0:
            k = min(TOP_K_CHUNKS, self.chunk_index.ntotal)
            scores, indices = self.chunk_index.search(q_emb, k)
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    hit = dict(self.chunks[idx])
                    hit["retrieval_score"] = float(score)
                    chunk_hits.append(hit)

        context_parts = []

        if topic_hits:
            context_parts.append("=== RELEVANT TOPIC SUMMARIES ===")
            for i, hit in enumerate(topic_hits, 1):
                context_parts.append(f"[Topic {i}] {hit['summary']}")

        if chunk_hits:
            context_parts.append("\n=== SUPPORTING CONVERSATION EXCERPTS ===")
            for i, hit in enumerate(chunk_hits, 1):
                context_parts.append(f"[Excerpt {i}]\n{hit['text']}")

        context_str = "\n".join(context_parts)
        return context_str, topic_hits, chunk_hits

    def save(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        if self.topic_index:
            faiss.write_index(self.topic_index, str(dir_path / "topic.index"))
        if self.chunk_index:
            faiss.write_index(self.chunk_index, str(dir_path / "chunk.index"))

        with open(dir_path / "topic_summaries.json", "w") as f:
            json.dump(self.topic_summaries, f)
        with open(dir_path / "chunks.json", "w") as f:
            json.dump(self.chunks, f)

        print(f"[retriever] Indices saved to {dir_path}")

    def load(self, dir_path: str | Path) -> bool:
        dir_path = Path(dir_path)
        topic_path = dir_path / "topic.index"
        chunk_path = dir_path / "chunk.index"

        if not topic_path.exists() or not chunk_path.exists():
            return False

        self.topic_index = faiss.read_index(str(topic_path))
        self.chunk_index = faiss.read_index(str(chunk_path))

        with open(dir_path / "topic_summaries.json") as f:
            self.topic_summaries = json.load(f)
        with open(dir_path / "chunks.json") as f:
            self.chunks = json.load(f)

        if self.topic_summaries:
            self._dim = len(self.topic_summaries[0]["embedding"])

        print(f"[retriever] Loaded indices from {dir_path}: "
              f"{self.topic_index.ntotal} topics, {self.chunk_index.ntotal} chunks")
        return True
