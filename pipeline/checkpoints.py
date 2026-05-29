"""
pipeline/checkpoints.py

Why 0.35?
   MiniLM cosine for related sentences: 0.5-0.9.
   Loosely related (tangents): 0.3-0.5.
   Unrelated topics: below 0.3.
   0.35 catches clear shifts without over-segmenting conversational tangents.
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
from collections import deque

import numpy as np
from google import genai
from google.genai import errors as genai_errors

from pipeline.loader import Message
from pipeline.embedder import embed, cosine_similarity, centroid

TOPIC_THRESHOLD    = float(os.getenv("TOPIC_SIMILARITY_THRESHOLD", "0.35"))
MIN_MSGS_BETWEEN   = int(os.getenv("MIN_MSGS_BETWEEN_CHECKPOINTS", "15"))
WINDOW_SIZE        = 6
STRUCTURAL_INTERVAL = 100

GEMINI_FAST_MODEL  = "gemini-2.0-flash"

@dataclass
class TopicCheckpoint:
    topic_id:  int
    start_msg: int
    end_msg:   int
    summary:   str
    embedding: List[float]

@dataclass
class StructuralCheckpoint:
    checkpoint_id: int
    msg_range: List[int]
    summary:   str
    embedding: List[float]

def _build_context(messages: List[Message]) -> str:
    return "\n".join(f"{m.speaker}: {m.text}" for m in messages)

def _summarise(messages: List[Message], client: genai.Client, retries: int = 5) -> str:
    context = _build_context(messages)
    prompt = (
        "Summarize this conversation segment in 3-4 sentences. "
        "Focus on what was being discussed, not who said it. Be concise and factual.\n\n"
        f"{context}"
    )
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_FAST_MODEL,
                contents=prompt,
            )
            return response.text.strip()
        except genai_errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 2 ** attempt * 15
                print(f"[checkpoints] Rate limited, waiting {wait}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini rate limit persisted after all retries")

def build_checkpoints(
    messages: List[Message],
    output_path: str | Path = "checkpoints.json",
    api_key: Optional[str] = None,
) -> dict:
    key = api_key or os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=key)

    topic_checkpoints: List[TopicCheckpoint] = []
    structural_checkpoints: List[StructuralCheckpoint] = []

    current_topic_start: int = 0
    current_topic_embeddings: List[np.ndarray] = []
    msgs_since_last_checkpoint: int = 0
    topic_id = 0
    rolling_window: deque = deque(maxlen=WINDOW_SIZE)

    print(f"[checkpoints] Processing {len(messages)} messages | threshold={TOPIC_THRESHOLD}")

    for i, msg in enumerate(messages):
        emb = embed(msg.text)[0]
        msg.embedding = emb.tolist()
        rolling_window.append(emb)
        current_topic_embeddings.append(emb)
        msgs_since_last_checkpoint += 1

        if (
            len(current_topic_embeddings) >= WINDOW_SIZE
            and msgs_since_last_checkpoint >= MIN_MSGS_BETWEEN
        ):
            topic_centroid = centroid(current_topic_embeddings)
            sim = cosine_similarity(emb, topic_centroid)

            if sim < TOPIC_THRESHOLD:
                segment = messages[current_topic_start:i]
                if segment:
                    print(f"[checkpoints] Topic change at msg {i} (sim={sim:.3f}), {len(segment)} msgs")
                    summary = _summarise(segment, client)
                    summary_emb = embed(summary)[0]
                    topic_checkpoints.append(TopicCheckpoint(
                        topic_id=topic_id,
                        start_msg=messages[current_topic_start].message_id,
                        end_msg=messages[i - 1].message_id,
                        summary=summary,
                        embedding=summary_emb.tolist(),
                    ))
                    topic_id += 1

                current_topic_start = i
                current_topic_embeddings = [emb]
                msgs_since_last_checkpoint = 0

        if (i + 1) % STRUCTURAL_INTERVAL == 0:
            start_i = max(0, i - STRUCTURAL_INTERVAL + 1)
            segment = messages[start_i:i + 1]
            print(f"[checkpoints] Structural checkpoint at msg {i}")
            summary = _summarise(segment, client)
            summary_emb = embed(summary)[0]
            structural_checkpoints.append(StructuralCheckpoint(
                checkpoint_id=len(structural_checkpoints),
                msg_range=[messages[start_i].message_id, messages[i].message_id],
                summary=summary,
                embedding=summary_emb.tolist(),
            ))

        if (i + 1) % 500 == 0:
            print(f"[checkpoints] {i+1}/{len(messages)} msgs | "
                  f"topics={len(topic_checkpoints)} structural={len(structural_checkpoints)}")

    remaining = messages[current_topic_start:]
    if remaining:
        print(f"[checkpoints] Final segment: {len(remaining)} msgs")
        summary = _summarise(remaining, client)
        summary_emb = embed(summary)[0]
        topic_checkpoints.append(TopicCheckpoint(
            topic_id=topic_id,
            start_msg=remaining[0].message_id,
            end_msg=remaining[-1].message_id,
            summary=summary,
            embedding=summary_emb.tolist(),
        ))

    result = {
        "topic_checkpoints": [asdict(tc) for tc in topic_checkpoints],
        "structural_checkpoints": [asdict(sc) for sc in structural_checkpoints],
        "stats": {
            "total_messages": len(messages),
            "total_topic_checkpoints": len(topic_checkpoints),
            "total_structural_checkpoints": len(structural_checkpoints),
        },
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[checkpoints] Saved → {output_path}")
    return result

def load_checkpoints(path: str | Path = "checkpoints.json") -> dict:
    with open(path) as f:
        return json.load(f)
