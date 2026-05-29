"""
build_pipeline.py

One-shot script to run the full pipeline locally.
Run this ONCE before starting the API server.
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from pipeline.loader import load_messages
from pipeline.checkpoints import build_checkpoints, load_checkpoints
from pipeline.retriever import Retriever
from pipeline.persona import build_persona, load_persona

def main():
    parser = argparse.ArgumentParser(description="Build the KaStack RAG + Persona pipeline")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--skip-persona", action="store_true")
    parser.add_argument("--skip-checkpoints", action="store_true")
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set. Add it to .env or pass --api-key")
        sys.exit(1)

    csv_path = BASE_DIR / "data" / "conversations.csv"
    print(f"\n{'='*60}")
    print("STEP 1: Loading messages from CSV")
    print(f"{'='*60}")
    messages = load_messages(csv_path, limit=1000)
    print(f"✓ {len(messages)} messages loaded\n")

    checkpoints_path = BASE_DIR / "checkpoints.json"
    if not args.skip_checkpoints:
        print(f"{'='*60}")
        print("STEP 2: Building topic + structural checkpoints")
        print(f"{'='*60}")
        if checkpoints_path.exists() and not args.rebuild:
            print(f"✓ checkpoints.json already exists — loading (use --rebuild to regenerate)")
            cp_data = load_checkpoints(checkpoints_path)
        else:
            cp_data = build_checkpoints(messages, checkpoints_path, api_key)
        print(f"✓ {len(cp_data['topic_checkpoints'])} topic checkpoints, "
              f"{len(cp_data['structural_checkpoints'])} structural\n")
    else:
        print("STEP 2: Skipped\n")
        cp_data = load_checkpoints(checkpoints_path) if checkpoints_path.exists() else {"topic_checkpoints": [], "structural_checkpoints": []}

    index_dir = BASE_DIR / "indices"
    print(f"{'='*60}")
    print("STEP 3: Building FAISS indices")
    print(f"{'='*60}")
    retriever = Retriever()
    if (index_dir / "topic.index").exists() and not args.rebuild:
        print(f"✓ Indices already exist — loading (use --rebuild to regenerate)")
        retriever.load(index_dir)
    else:
        retriever.build_topic_index(cp_data.get("topic_checkpoints", []))
        retriever.build_chunk_index(messages)
        retriever.save(index_dir)
    print("✓ FAISS indices ready\n")

    persona_path = BASE_DIR / "persona.json"
    if not args.skip_persona:
        print(f"{'='*60}")
        print("STEP 4: Extracting persona (3-pass)")
        print(f"{'='*60}")
        if persona_path.exists() and not args.rebuild:
            print(f"✓ persona.json already exists — loading (use --rebuild to regenerate)")
        else:
            build_persona(messages, persona_path, api_key)
        print("✓ Persona ready\n")
    else:
        print("STEP 4: Skipped\n")

    print(f"{'='*60}")
    print("✓ PIPELINE BUILD COMPLETE")
    print(f"{'='*60}")
    print(f"  checkpoints.json  → {checkpoints_path}")
    print(f"  persona.json      → {persona_path}")
    print(f"  indices/          → {index_dir}")
    print("\nNext: uvicorn api.main:app --reload\n")

if __name__ == "__main__":
    main()
