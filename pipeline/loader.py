import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class Message:
    message_id: int
    conversation_id: int
    turn_index: int
    speaker: str
    text: str
    embedding: Optional[List[float]] = field(default=None, repr=False)

_SPEAKER_RE = re.compile(r'^([\w\s]+\d*):\s+', re.IGNORECASE)

def _parse_conversation(raw_text: str, conversation_id: int, start_message_id: int) -> List[Message]:
    messages: List[Message] = []
    current_speaker: Optional[str] = None
    current_text_parts: List[str] = []
    turn_index = 0
    msg_id = start_message_id

    def flush_current():
        nonlocal turn_index, msg_id
        if current_speaker and current_text_parts:
            text = " ".join(current_text_parts).strip()
            if text:
                messages.append(Message(
                    message_id=msg_id,
                    conversation_id=conversation_id,
                    turn_index=turn_index,
                    speaker=current_speaker,
                    text=text,
                ))
                msg_id += 1
                turn_index += 1

    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = _SPEAKER_RE.match(line)
        if match:
            flush_current()
            current_speaker = match.group(1).strip()
            current_text_parts = [line[match.end():].strip()]
        else:
            if current_speaker:
                current_text_parts.append(line)

    flush_current()
    return messages, msg_id

def load_messages(csv_path: str | Path, limit: int = None) -> List[Message]:
    """
    CSV Format Note:
    Each row contains a FULL multi-turn conversation as a single cell.
    (conversation_index, turn_index) preserves chronological intent.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    all_messages: List[Message] = []
    global_msg_id = 0

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for conv_id, row in enumerate(reader):
            if not row or not row[0].strip():
                continue
            raw_text = row[0]
            msgs, global_msg_id = _parse_conversation(raw_text, conv_id, global_msg_id)
            all_messages.extend(msgs)
            if limit and len(all_messages) >= limit:
                all_messages = all_messages[:limit]
                break

    all_messages.sort(key=lambda m: (m.conversation_id, m.turn_index))
    print(f"[loader] Loaded {len(all_messages)} messages from {len(set(m.conversation_id for m in all_messages))} conversations.")
    return all_messages
