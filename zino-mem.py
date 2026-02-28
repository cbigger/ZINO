#!/usr/bin/env python3
"""
zino-mem — memory service.

Responsibilities:
  - Stores and retrieves channel-based chat histories.
  - Stores and retrieves hard memories (user:assistant exchanges)
    with similarity-based search (TF-IDF cosine similarity).
  - Stores and retrieves the soft memory string.

UDS socket: /run/zino/mem.sock (configurable)

Inbound message types:
  {"type": "get_history", "channel_id": str, "limit": int (optional)}
      → {"type": "history", "messages": [...]}

  {"type": "append_history", "channel_id": str, "messages": [...]}
      → {"type": "ok"}

  {"type": "clear_history", "channel_id": str}
      → {"type": "ok"}

  {"type": "search_hard", "query": str, "top_k": int (optional, default 3)}
      → {"type": "hard_memories", "results": [{"messages": [...], "score": float}, ...]}

  {"type": "store_hard", "messages": [...], "tag": str (optional)}
      → {"type": "ok", "id": int}

  {"type": "get_soft"}
      → {"type": "soft_memory", "content": str}

  {"type": "set_soft", "content": str}
      → {"type": "ok"}

  {"type": "ping"}
      → {"type": "pong"}
"""

import asyncio
import json
import math
import os
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path

from zino_common import send_msg, recv_msg, setup_logging

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[mem] Config not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Simple TF-IDF similarity search
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def compute_tf(tokens: list[str]) -> dict[str, float]:
    """Term frequency: count / total."""
    counts = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {t: c / total for t, c in counts.items()}


def cosine_sim(v1: dict[str, float], v2: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    common = set(v1.keys()) & set(v2.keys())
    if not common:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in common)
    mag1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


# ---------------------------------------------------------------------------
# Memory service state
# ---------------------------------------------------------------------------


class MemService:
    def __init__(self, config: dict):
        self.config = config
        mem_cfg = config.get("mem", {})
        self.data_dir = Path(mem_cfg.get("data_dir", "data/mem"))
        self.channels_dir = self.data_dir / "channels"
        self.hard_path = self.data_dir / "hard_memories.json"
        self.soft_path = self.data_dir / "soft_memory.txt"

        self.channels_dir.mkdir(parents=True, exist_ok=True)

        # Load hard memories
        self.hard_memories: list[dict] = []
        if self.hard_path.exists():
            with open(self.hard_path, "r", encoding="utf-8") as f:
                self.hard_memories = json.load(f)
        log.info("loaded %d hard memory entries.", len(self.hard_memories))

        # Load soft memory
        self.soft_memory = ""
        if self.soft_path.exists():
            self.soft_memory = self.soft_path.read_text(encoding="utf-8")

        log.info("data directory: %s", self.data_dir)

    # -- Chat history --

    def _channel_path(self, channel_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", channel_id)
        return self.channels_dir / f"{safe_id}.json"

    def get_history(self, channel_id: str, limit: int = 0) -> list[dict]:
        path = self._channel_path(channel_id)
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            messages = json.load(f)
        if limit > 0:
            messages = messages[-limit:]
        return messages

    def append_history(self, channel_id: str, messages: list[dict]):
        path = self._channel_path(channel_id)
        existing = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing.extend(messages)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    def clear_history(self, channel_id: str):
        path = self._channel_path(channel_id)
        if path.exists():
            path.unlink()

    # -- Hard memories --

    def _save_hard(self):
        self.hard_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.hard_path, "w", encoding="utf-8") as f:
            json.dump(self.hard_memories, f, ensure_ascii=False, indent=2)

    def _memory_text(self, entry: dict) -> str:
        """Extract searchable text from a hard memory entry."""
        parts = []
        for msg in entry.get("messages", []):
            parts.append(msg.get("content", ""))
        if entry.get("tag"):
            parts.append(entry["tag"])
        return " ".join(parts)

    def store_hard(self, messages: list[dict], tag: str = "") -> int:
        entry = {"messages": messages, "tag": tag}
        self.hard_memories.append(entry)
        self._save_hard()
        return len(self.hard_memories) - 1

    def search_hard(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.hard_memories or not query.strip():
            return []

        query_tokens = tokenize(query)
        query_tf = compute_tf(query_tokens)

        all_docs = [tokenize(self._memory_text(m)) for m in self.hard_memories]
        n_docs = len(all_docs) + 1

        term_doc_count: Counter = Counter()
        for doc_tokens in all_docs:
            for term in set(doc_tokens):
                term_doc_count[term] += 1
        for term in set(query_tokens):
            term_doc_count[term] += 1

        idf = {t: math.log(n_docs / (1 + c)) for t, c in term_doc_count.items()}

        query_vec = {t: tf * idf.get(t, 0) for t, tf in query_tf.items()}

        scored = []
        for i, doc_tokens in enumerate(all_docs):
            doc_tf = compute_tf(doc_tokens)
            doc_vec = {t: tf * idf.get(t, 0) for t, tf in doc_tf.items()}
            score = cosine_sim(query_vec, doc_vec)
            if score > 0:
                scored.append((score, i))

        scored.sort(reverse=True)
        results = []
        for score, idx in scored[:top_k]:
            results.append({
                "messages": self.hard_memories[idx]["messages"],
                "score": round(score, 4),
            })
        return results

    # -- Soft memory --

    def get_soft(self) -> str:
        return self.soft_memory

    def set_soft(self, content: str):
        self.soft_memory = content
        self.soft_path.parent.mkdir(parents=True, exist_ok=True)
        self.soft_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Connection handler
# ---------------------------------------------------------------------------


async def handle_connection(
    service: MemService,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
):
    try:
        msg = await recv_msg(reader)
        msg_type = msg.get("type")

        log.info("request type=%s", msg_type)

        if msg_type == "ping":
            await send_msg(writer, {"type": "pong"})

        elif msg_type == "get_history":
            channel_id = msg.get("channel_id", "")
            limit = int(msg.get("limit", 0))
            if not channel_id:
                await send_msg(writer, {"type": "error",
                                        "message": "channel_id required"})
                return
            messages = service.get_history(channel_id, limit)
            log.info("get_history: channel=%s limit=%d returned=%d",
                     channel_id, limit, len(messages))
            await send_msg(writer, {"type": "history", "messages": messages})

        elif msg_type == "append_history":
            channel_id = msg.get("channel_id", "")
            messages = msg.get("messages", [])
            if not channel_id:
                await send_msg(writer, {"type": "error",
                                        "message": "channel_id required"})
                return
            service.append_history(channel_id, messages)
            log.info("append_history: channel=%s messages=%d", channel_id, len(messages))
            await send_msg(writer, {"type": "ok"})

        elif msg_type == "clear_history":
            channel_id = msg.get("channel_id", "")
            if not channel_id:
                await send_msg(writer, {"type": "error",
                                        "message": "channel_id required"})
                return
            service.clear_history(channel_id)
            log.info("clear_history: channel=%s", channel_id)
            await send_msg(writer, {"type": "ok"})

        elif msg_type == "search_hard":
            query = msg.get("query", "")
            top_k = int(msg.get("top_k", 3))
            results = service.search_hard(query, top_k)
            log.info("search_hard: query_len=%d top_k=%d results=%d",
                     len(query), top_k, len(results))
            log.debug("search_hard query: %s", query)
            await send_msg(writer, {"type": "hard_memories", "results": results})

        elif msg_type == "store_hard":
            messages = msg.get("messages", [])
            tag = msg.get("tag", "")
            idx = service.store_hard(messages, tag)
            log.info("store_hard: id=%d messages=%d tag=%s", idx, len(messages), tag)
            await send_msg(writer, {"type": "ok", "id": idx})

        elif msg_type == "get_soft":
            log.info("get_soft: %d chars", len(service.get_soft()))
            await send_msg(writer, {"type": "soft_memory",
                                    "content": service.get_soft()})

        elif msg_type == "set_soft":
            content = msg.get("content", "")
            service.set_soft(content)
            log.info("set_soft: %d chars", len(content))
            await send_msg(writer, {"type": "ok"})

        else:
            await send_msg(writer, {"type": "error",
                                    "message": f"Unknown message type: {msg_type}"})
            log.warning("unknown message type: %s", msg_type)

    except asyncio.IncompleteReadError:
        pass
    except Exception as e:
        log.error("handler error: %s", e)
    finally:
        writer.close()
        await writer.wait_closed()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(config_path: str):
    config = load_config(config_path)

    global log
    log = setup_logging("zino.mem", config)

    service = MemService(config)

    socket_path = config.get("mem", {}).get("socket", "/run/zino/mem.sock")
    Path(socket_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        Path(socket_path).unlink()
    except FileNotFoundError:
        pass

    server = await asyncio.start_unix_server(
        lambda r, w: handle_connection(service, r, w),
        path=socket_path,
    )

    log.info("listening on %s", socket_path)
    async with server:
        await server.serve_forever()


log = None  # set in main()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="zino-mem: memory service")
    parser.add_argument("--config", "-c",
                        default=os.environ.get("ZINO_CONFIG", "ZINO.toml"))
    args = parser.parse_args()
    asyncio.run(main(args.config))
