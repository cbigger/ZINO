#!/usr/bin/env python3
"""
zino-ctx — context, memory, and system prompt service.

Combines the functionality of the former zino-mem, zino-sys, and zino-ctx
services into a single process.

Responsibilities:
  - Stores and retrieves channel-based chat histories (JSON files on disk).
  - Stores and retrieves hard memories with TF-IDF similarity search.
  - Stores and retrieves the soft memory string.
  - Loads and renders the system prompt template (SYSTEM.md).
  - Assembles context messages (skill examples, hard memories, chat history).

UDS socket: /run/zino/ctx.sock (configurable)

Inbound message types:

  Memory — chat history:
    {"type": "get_history", "channel_id": str, "limit": int?}
        → {"type": "history", "messages": [...]}
    {"type": "append_history", "channel_id": str, "messages": [...]}
        → {"type": "ok"}
    {"type": "clear_history", "channel_id": str}
        → {"type": "ok"}

  Memory — hard memories:
    {"type": "search_hard", "query": str, "top_k": int?}
        → {"type": "hard_memories", "results": [{"messages": [...], "score": float}, ...]}
    {"type": "store_hard", "messages": [...], "tag": str?}
        → {"type": "ok", "id": int}

  Memory — soft memory:
    {"type": "get_soft"}
        → {"type": "soft_memory", "content": str}
    {"type": "set_soft", "content": str}
        → {"type": "ok"}

  System prompt:
    {"type": "get"}
        → {"type": "system_prompt", "content": str}
    {"type": "reload"}
        → {"type": "system_prompt", "content": str}
    {"type": "set_soft_memory", "content": str}
        → {"type": "system_prompt", "content": str}

  Context assembly:
    {"type": "build", "channel_id": str?, "user_message": str}
        → {"type": "context", "messages": [...]}

  Health:
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
        print(f"[ctx] Config not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# TF-IDF similarity helpers
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
# MemService — persistent memory (chat history, hard memories, soft memory)
# ---------------------------------------------------------------------------


class MemService:
    def __init__(self, config: dict):
        ctx_cfg = config.get("ctx", {})
        self.data_dir = Path(ctx_cfg.get("data_dir", "data/mem"))
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
# Tool and skill loading (from former zino-sys)
# ---------------------------------------------------------------------------


def load_tools(config: dict) -> dict[str, dict]:
    tools_dir = Path(config.get("tools", {}).get("dir", "tools"))
    if not tools_dir.exists():
        log.warning("tools directory not found: %s — no tools loaded.", tools_dir)
        return {}
    tools: dict[str, dict] = {}
    for tool_file in sorted(tools_dir.glob("*.json")):
        with open(tool_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        tools[tool_file.stem] = {"meta": data.get("meta", {})}
    return tools


def load_skills(config: dict) -> dict[str, dict]:
    skills_dir = Path(config.get("skills", {}).get("dir", "skills"))
    if not skills_dir.exists():
        log.warning("skills directory not found: %s — no skills loaded.", skills_dir)
        return {}
    skills: dict[str, dict] = {}
    for skill_file in sorted(skills_dir.glob("*.json")):
        with open(skill_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        skills[skill_file.stem] = {
            "meta":     data.get("meta", {}),
            "system":   data.get("system", ""),
            "tools":    data.get("tools", []),
            "examples": data.get("examples", []),
        }
    return skills


# ---------------------------------------------------------------------------
# System prompt builder (from former zino-sys)
# ---------------------------------------------------------------------------


def load_template(config: dict) -> str:
    path = config.get("ctx", {}).get("template", "SYSTEM.md")
    p = Path(path)
    if not p.exists():
        log.warning("SYSTEM.md not found at %s — serving empty system prompt.", path)
        return ""
    return p.read_text(encoding="utf-8")


def _tool_fn_name(executor: str) -> str:
    return f"execute_{executor}"


def build_prompt(config: dict, tools: dict, skills: dict, soft_memory: str) -> str:
    template = load_template(config)
    if not template:
        return ""

    # %%PERSONALITY%%
    personality = config.get("ctx", {}).get("personality", "")

    # %%TOOL_LIST%%
    seen_executors: set[str] = set()
    tool_lines: list[str] = []
    for tool_meta in tools.values():
        executor = tool_meta["meta"].get("executor", "bash")
        if executor not in seen_executors:
            seen_executors.add(executor)
            tool_lines.append(f'  - "{_tool_fn_name(executor)}": {executor} executor')
    tool_list = "\n".join(tool_lines) if tool_lines else "  (none loaded)"

    # %%SKILL_LINES%%
    skill_lines = "\n".join(
        f'  - "{name}": {s["meta"].get("executor", "unknown")} executor, '
        f'static analysis: {s["meta"].get("static_analysis") or "none"}'
        for name, s in skills.items()
    ) if skills else "  (none loaded)"

    # %%MAX_ITERATIONS%%
    max_iterations = str(config.get("agent", {}).get("max_iterations", 5))

    return (
        template
        .replace("%%PERSONALITY%%",    personality)
        .replace("%%TOOL_LIST%%",      tool_list)
        .replace("%%SKILL_LINES%%",    skill_lines)
        .replace("%%MAX_ITERATIONS%%", max_iterations)
        .replace("%%SOFT_MEMORIES%%",  soft_memory)
    )


class SysService:
    def __init__(self, config: dict, mem: MemService):
        self.config      = config
        self.mem         = mem
        self.tools       = load_tools(config)
        self.skills      = load_skills(config)
        # Use soft memory from mem (which loaded from disk), falling back to config
        if not self.mem.soft_memory:
            initial = config.get("ctx", {}).get("soft_memory", "")
            if initial:
                self.mem.set_soft(initial)
        self._prompt     = build_prompt(config, self.tools, self.skills, self.mem.get_soft())
        log.info("loaded %d tool(s), %d skill(s).", len(self.tools), len(self.skills))

    @property
    def prompt(self) -> str:
        return self._prompt

    def reload(self) -> str:
        self.tools   = load_tools(self.config)
        self.skills  = load_skills(self.config)
        self._prompt = build_prompt(self.config, self.tools, self.skills, self.mem.get_soft())
        log.info("reloaded: %d tool(s), %d skill(s).", len(self.tools), len(self.skills))
        return self._prompt

    def set_soft_memory(self, content: str) -> str:
        self.mem.set_soft(content)
        self._prompt = build_prompt(self.config, self.tools, self.skills, content)
        log.info("soft memory updated, prompt rebuilt.")
        return self._prompt


# ---------------------------------------------------------------------------
# Skill example loading (from former zino-ctx)
# ---------------------------------------------------------------------------


def load_skill_examples(config: dict) -> list[dict]:
    """
    Load example exchanges from skill definitions (fabricator examples).
    Skips the system message; takes only user:assistant pairs.
    """
    skills_dir = Path(config.get("skills", {}).get("dir", "skills"))
    examples = []
    if not skills_dir.exists():
        return examples

    for skill_file in sorted(skills_dir.glob("*.json")):
        with open(skill_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for msg in data.get("fabricator", []):
            if msg.get("role") in ("user", "assistant"):
                examples.append(msg)

    return examples


# ---------------------------------------------------------------------------
# Context assembly (from former zino-ctx)
# ---------------------------------------------------------------------------


class CtxService:
    def __init__(self, config: dict, mem: MemService):
        self.config = config
        self.mem = mem
        ctx_cfg = config.get("ctx", {})
        self.skill_examples = load_skill_examples(config)
        self.history_limit = ctx_cfg.get("history_limit", 50)
        self.hard_memory_top_k = ctx_cfg.get("hard_memory_top_k", 3)
        self.inject_history = ctx_cfg.get("inject_history", True)
        self.inject_hard_memories = ctx_cfg.get("inject_hard_memories", False)
        self.inject_skill_examples = ctx_cfg.get("inject_skill_examples", False)
        log.info("loaded %d skill example messages.", len(self.skill_examples))
        log.info("inject flags: history=%s hard_memories=%s skill_examples=%s",
                 self.inject_history, self.inject_hard_memories,
                 self.inject_skill_examples)

    def build_context(
        self, channel_id: str | None, user_message: str,
    ) -> list[dict]:
        """
        Build the context messages list.  Order:
          1. Skill/tool examples (static, loaded at startup)
          2. Hard memories (similarity search on user_message)
          3. Chat history (by channel_id)
        """
        messages = []

        # 1. Skill/tool examples (opt-in)
        example_count = 0
        if self.inject_skill_examples:
            messages.extend(self.skill_examples)
            example_count = len(self.skill_examples)

        # 2. Hard memories (opt-in)
        hard_count = 0
        if self.inject_hard_memories and user_message.strip():
            results = self.mem.search_hard(user_message, self.hard_memory_top_k)
            for r in results:
                messages.extend(r.get("messages", []))
                hard_count += len(r.get("messages", []))

        # 3. Chat history (default on)
        history_count = 0
        if self.inject_history and channel_id:
            history = self.mem.get_history(channel_id, self.history_limit)
            messages.extend(history)
            history_count = len(history)

        log.info("built context: %d examples + %d hard + %d history = %d total",
                 example_count, hard_count, history_count, len(messages))
        return messages


# ---------------------------------------------------------------------------
# Connection handler — dispatches all message types
# ---------------------------------------------------------------------------


async def handle_connection(
    mem: MemService,
    sys_svc: SysService,
    ctx: CtxService,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
):
    try:
        msg = await recv_msg(reader)
        msg_type = msg.get("type")

        log.info("request type=%s", msg_type)

        # -- Health --

        if msg_type == "ping":
            await send_msg(writer, {"type": "pong"})

        # -- Memory: chat history --

        elif msg_type == "get_history":
            channel_id = msg.get("channel_id", "")
            limit = int(msg.get("limit", 0))
            if not channel_id:
                await send_msg(writer, {"type": "error",
                                        "message": "channel_id required"})
                return
            messages = mem.get_history(channel_id, limit)
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
            mem.append_history(channel_id, messages)
            log.info("append_history: channel=%s messages=%d", channel_id, len(messages))
            await send_msg(writer, {"type": "ok"})

        elif msg_type == "clear_history":
            channel_id = msg.get("channel_id", "")
            if not channel_id:
                await send_msg(writer, {"type": "error",
                                        "message": "channel_id required"})
                return
            mem.clear_history(channel_id)
            log.info("clear_history: channel=%s", channel_id)
            await send_msg(writer, {"type": "ok"})

        # -- Memory: hard memories --

        elif msg_type == "search_hard":
            query = msg.get("query", "")
            top_k = int(msg.get("top_k", 3))
            results = mem.search_hard(query, top_k)
            log.info("search_hard: query_len=%d top_k=%d results=%d",
                     len(query), top_k, len(results))
            log.debug("search_hard query: %s", query)
            await send_msg(writer, {"type": "hard_memories", "results": results})

        elif msg_type == "store_hard":
            messages = msg.get("messages", [])
            tag = msg.get("tag", "")
            idx = mem.store_hard(messages, tag)
            log.info("store_hard: id=%d messages=%d tag=%s", idx, len(messages), tag)
            await send_msg(writer, {"type": "ok", "id": idx})

        # -- Memory: soft memory --

        elif msg_type == "get_soft":
            log.info("get_soft: %d chars", len(mem.get_soft()))
            await send_msg(writer, {"type": "soft_memory",
                                    "content": mem.get_soft()})

        elif msg_type == "set_soft":
            content = msg.get("content", "")
            # Persist soft memory and rebuild system prompt
            sys_svc.set_soft_memory(content)
            log.info("set_soft: %d chars (prompt rebuilt)", len(content))
            await send_msg(writer, {"type": "ok"})

        # -- System prompt --

        elif msg_type == "get":
            await send_msg(writer, {"type": "system_prompt", "content": sys_svc.prompt})
            log.debug("served system prompt (%d chars)", len(sys_svc.prompt))

        elif msg_type == "reload":
            prompt = sys_svc.reload()
            await send_msg(writer, {"type": "system_prompt", "content": prompt})

        elif msg_type == "set_soft_memory":
            content = msg.get("content", "")
            prompt = sys_svc.set_soft_memory(content)
            await send_msg(writer, {"type": "system_prompt", "content": prompt})

        # -- Context assembly --

        elif msg_type == "build":
            channel_id = msg.get("channel_id")
            user_message = msg.get("user_message", "")
            log.info("build: channel=%s msg_len=%d", channel_id, len(user_message))
            context = ctx.build_context(channel_id, user_message)
            await send_msg(writer, {"type": "context", "messages": context})

        # -- Unknown --

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
    log = setup_logging("zino.ctx", config)

    mem = MemService(config)
    sys_svc = SysService(config, mem)
    ctx = CtxService(config, mem)

    # Seed skill context channels from skill example definitions
    for name, skill in sys_svc.skills.items():
        examples = skill.get("examples", [])
        channel = f"_skill_{name}"
        mem.clear_history(channel)
        if examples:
            mem.append_history(channel, examples)
        log.info("seeded skill channel %s with %d example messages.", channel, len(examples))

    socket_path = config.get("ctx", {}).get("socket", "/run/zino/ctx.sock")
    Path(socket_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        Path(socket_path).unlink()
    except FileNotFoundError:
        pass

    server = await asyncio.start_unix_server(
        lambda r, w: handle_connection(mem, sys_svc, ctx, r, w),
        path=socket_path,
    )

    log.info("listening on %s", socket_path)
    async with server:
        await server.serve_forever()


log = None  # set in main()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="zino-ctx: context, memory, and system prompt service")
    parser.add_argument("--config", "-c",
                        default=os.environ.get("ZINO_CONFIG", "ZINO.toml"))
    args = parser.parse_args()
    asyncio.run(main(args.config))
