"""
pipeline/persona.py

Multi-pass persona extraction using Google Gemini.

Pass 1 — Facts (batches of 50 messages)
Pass 2 — Habits (batches of 50 messages)
Pass 3a — Communication style (programmatic, no LLM)
Pass 3b — Personality (100-message sample via Gemini)
"""

import json
import os
import re
import random
import statistics
import time
from pathlib import Path
from typing import List, Optional

from google import genai
from google.genai import errors as genai_errors
from tqdm import tqdm

from pipeline.loader import Message

BATCH_SIZE         = 50
PERSONALITY_SAMPLE = 100
GEMINI_FAST_MODEL  = "gemini-2.0-flash"

_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001FA00-\U0001FA6F\u2600-\u27BF]+",
    flags=re.UNICODE,
)

_SLANG_TERMS = {
    "lol","lmao","omg","omfg","wtf","brb","imo","imho","irl","ngl","tbh",
    "rn","afk","gtg","gg","ikr","smh","fwiw","afaik","idk","thx","ty","np",
    "pls","plz","bc","cuz","ur","u","r","wat","wut","ya","yea","nah",
    "gonna","wanna","gotta","lemme","kinda","sorta","dunno",
}

def _batch_messages(messages: List[Message], size: int) -> List[List[Message]]:
    return [messages[i:i+size] for i in range(0, len(messages), size)]

def _fmt(batch: List[Message]) -> str:
    return "\n".join(f"{m.speaker}: {m.text}" for m in batch)

def _call_json(prompt: str, client: genai.Client, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_FAST_MODEL,
                contents=prompt,
            )
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except genai_errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 2 ** attempt * 15
                print(f"[persona] Rate limited, waiting {wait}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait)
            else:
                return {}
        except Exception:
            return {}
    return {}

def _merge(*lists) -> list:
    seen, result = set(), []
    for lst in lists:
        for item in (lst or []):
            key = str(item).lower().strip()
            if key and key not in seen:
                seen.add(key)
                result.append(item)
    return result

def extract_facts(messages: List[Message], client: genai.Client) -> dict:
    print(f"[persona] Pass 1 — Facts ({len(messages)} msgs, batch={BATCH_SIZE})")
    rels, locs, evts = [], [], []
    for batch in tqdm(_batch_messages(messages, BATCH_SIZE), desc="Facts"):
        prompt = (
            "From these messages, extract only explicitly mentioned personal facts. "
            'Return ONLY JSON: {"relationships": [], "locations": [], "events": []}\n'
            "relationships=named people+relation, locations=specific places, events=specific events. "
            "Empty list if nothing found. No explanations.\n\n"
            f"{_fmt(batch)}"
        )
        r = _call_json(prompt, client)
        rels.extend(r.get("relationships", []))
        locs.extend(r.get("locations", []))
        evts.extend(r.get("events", []))
    return {"relationships": _merge(rels), "locations": _merge(locs), "events": _merge(evts)}

def extract_habits(messages: List[Message], client: genai.Client) -> dict:
    print(f"[persona] Pass 2 — Habits ({len(messages)} msgs, batch={BATCH_SIZE})")
    sleep_all, food_all, routines_all = [], [], []
    for batch in tqdm(_batch_messages(messages, BATCH_SIZE), desc="Habits"):
        prompt = (
            "Identify behavioral patterns in these messages. "
            'Return ONLY JSON: {"sleep": [], "food": [], "routines": []}\n'
            "sleep=sleep times/habits, food=foods/diets/restaurants, routines=recurring activities. "
            "Empty list if nothing found. No explanations.\n\n"
            f"{_fmt(batch)}"
        )
        r = _call_json(prompt, client)
        sv = r.get("sleep", [])
        sleep_all.extend(sv if isinstance(sv, list) else ([sv] if sv else []))
        food_all.extend(r.get("food", []))
        routines_all.extend(r.get("routines", []))
    return {
        "sleep": "; ".join(_merge(sleep_all)) if sleep_all else "Not mentioned",
        "food": _merge(food_all),
        "routines": _merge(routines_all),
    }

def analyse_communication_style(messages: List[Message]) -> dict:
    lengths      = [len(m.text) for m in messages]
    words        = [len(m.text.split()) for m in messages]
    emojis       = [len(_EMOJI_RE.findall(m.text)) for m in messages]
    questions    = [1 if "?" in m.text else 0 for m in messages]
    slang_flags  = [1 if any(w.lower() in _SLANG_TERMS for w in m.text.split()) else 0 for m in messages]

    n            = len(messages)
    avg_len      = statistics.mean(lengths) if lengths else 0
    std_len      = statistics.stdev(lengths) if n > 1 else 0
    avg_words    = statistics.mean(words) if words else 0
    avg_emojis   = sum(emojis) / n if n else 0
    q_ratio      = sum(questions) / n if n else 0
    slang_ratio  = sum(slang_flags) / n if n else 0

    emoji_class  = "high" if avg_emojis > 0.3 else ("medium" if avg_emojis > 0.1 else "low")

    patterns = []
    if q_ratio > 0.4:       patterns.append("Frequently asks questions (inquisitive)")
    if slang_ratio > 0.3:   patterns.append("Heavy slang/abbreviation use (casual)")
    if avg_words < 10:      patterns.append("Short, punchy messages")
    elif avg_words > 40:    patterns.append("Writes long, detailed messages")
    if std_len > avg_len * 0.8: patterns.append("Message length varies widely")

    return {
        "avg_message_length":   round(avg_len, 1),
        "avg_words_per_message": round(avg_words, 1),
        "std_message_length":   round(std_len, 1),
        "emoji_usage":          emoji_class,
        "question_ratio":       round(q_ratio, 3),
        "slang_ratio":          round(slang_ratio, 3),
        "notable_patterns":     patterns,
        "tone": "",
    }

def extract_personality(messages: List[Message], client: genai.Client) -> tuple:
    sample = random.sample(messages, min(PERSONALITY_SAMPLE, len(messages)))
    prompt = (
        "Based on these conversation samples, identify personality traits. "
        'Return ONLY JSON: {"traits": [], "tone": ""}\n'
        "traits=5-8 personality descriptors, tone=one sentence. No explanations.\n\n"
        f"{_fmt(sample)}"
    )
    r = _call_json(prompt, client)
    return (
        r.get("traits", []),
        r.get("tone", "Conversational and friendly"),
    )

def build_persona(
    messages: List[Message],
    output_path: str | Path = "persona.json",
    api_key: Optional[str] = None,
) -> dict:
    key    = api_key or os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=key)

    MAX_LLM = 5000
    llm_msgs = messages[:MAX_LLM] if len(messages) > MAX_LLM else messages
    print(f"[persona] LLM passes on {len(llm_msgs)} msgs (total {len(messages)})")

    facts  = extract_facts(llm_msgs, client)
    habits = extract_habits(llm_msgs, client)
    style  = analyse_communication_style(messages)
    traits, tone = extract_personality(messages, client)
    style["tone"] = tone

    persona = {
        "facts": facts,
        "habits": habits,
        "personality": traits,
        "communication_style": style,
        "_meta": {
            "total_messages_analysed": len(messages),
            "llm_messages_sample": len(llm_msgs),
            "passes": ["facts", "habits", "communication_stats", "personality"],
            "llm": GEMINI_FAST_MODEL,
        },
    }
    with open(output_path, "w") as f:
        json.dump(persona, f, indent=2)
    print(f"[persona] Saved → {output_path}")
    return persona

def load_persona(path: str | Path = "persona.json") -> dict:
    with open(path) as f:
        return json.load(f)

def persona_to_paragraph(persona: dict) -> str:
    facts  = persona.get("facts", {})
    habits = persona.get("habits", {})
    comm   = persona.get("communication_style", {})
    traits = persona.get("personality", [])

    locs     = ", ".join(facts.get("locations", [])[:5]) or "various places"
    food     = ", ".join(habits.get("food", [])[:3]) or "various foods"
    routines = "; ".join(habits.get("routines", [])[:3]) or "unspecified routines"
    p_str    = ", ".join(traits[:5]) or "friendly and conversational"
    tone     = comm.get("tone", "Conversational and friendly")
    emoji    = comm.get("emoji_usage", "medium")
    avg_len  = comm.get("avg_message_length", "N/A")

    return (
        f"The person in this conversation is {p_str}. "
        f"They mention places like {locs} and enjoy {food}. "
        f"Regular activities: {routines}. "
        f"Communication style: {tone.lower().rstrip('.')} with {emoji} emoji usage "
        f"and ~{avg_len} character average message length."
    )
