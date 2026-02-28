#!/usr/bin/env python3
"""
zino-ctx — context history service.

Responsibilities:
  - Assembles the context history for LLM API calls.
  - Builds example exchanges from skill definitions (fabricator examples).
  - Retrieves hard memories from zino-mem (similarity search on user message).
  - Retrieves chat history from zino-mem (by channel_id).
  - Returns the assembled messages list.

The returned messages are inserted between the system prompt and the
current user message in the final messages array.

UDS socket: /run/zino/ctx.sock (configurable)

Inbound message types:
  {"type": "build", "channel_id": str|null, "user_message": str}
      → {"type": "context", "messages": [...]}

  {"type": "ping"}
      → {"type": "pong"}
"""

import asyncio
import json
import os
import sys
import tomllib
from pathlib import Path

from zino_common import send_msg, recv_msg, open_uds, setup_logging

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
# Skill example loading
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
# Memory service interaction
# ---------------------------------------------------------------------------


async def fetch_chat_history(
    mem_socket: str, channel_id: str, limit: int = 50,
) -> list[dict]:
    """Retrieve chat history from zino-mem."""
    try:
        reader, writer = await open_uds(mem_socket)
        await send_msg(writer, {
            "type": "get_history",
            "channel_id": channel_id,
            "limit": limit,
        })
        resp = await recv_msg(reader)
        writer.close()
        await writer.wait_closed()
        if resp.get("type") == "history":
            return resp.get("messages", [])
        return []
    except Exception as e:
        log.warning("could not fetch history from mem: %s", e)
        return []


async def search_hard_memories(
    mem_socket: str, query: str, top_k: int = 3,
) -> list[dict]:
    """Search hard memories from zino-mem by similarity."""
    try:
        reader, writer = await open_uds(mem_socket)
        await send_msg(writer, {
            "type": "search_hard",
            "query": query,
            "top_k": top_k,
        })
        resp = await recv_msg(reader)
        writer.close()
        await writer.wait_closed()
        if resp.get("type") == "hard_memories":
            messages = []
            for r in resp.get("results", []):
                messages.extend(r.get("messages", []))
            return messages
        return []
    except Exception as e:
        log.warning("could not search hard memories from mem: %s", e)
        return []


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


class CtxService:
    def __init__(self, config: dict):
        self.config = config
        self.mem_socket = config.get("mem", {}).get("socket", "/run/zino/mem.sock")
        self.skill_examples = load_skill_examples(config)
        self.history_limit = config.get("ctx", {}).get("history_limit", 50)
        self.hard_memory_top_k = config.get("ctx", {}).get("hard_memory_top_k", 3)
        log.info("loaded %d skill example messages.", len(self.skill_examples))

    async def build_context(
        self, channel_id: str | None, user_message: str,
    ) -> list[dict]:
        """
        Build the context messages list.  Order per README spec:
          1. Skill/tool examples (static, loaded at startup)
          2. Hard memories (similarity search on user_message via zino-mem)
          3. Chat history (from zino-mem, by channel_id)
        """
        messages = []

        # 1. Skill/tool examples
        messages.extend(self.skill_examples)

        # 2. Hard memories
        hard_count = 0
        if user_message.strip():
            hard = await search_hard_memories(
                self.mem_socket, user_message, self.hard_memory_top_k,
            )
            messages.extend(hard)
            hard_count = len(hard)

        # 3. Chat history
        history_count = 0
        if channel_id:
            history = await fetch_chat_history(
                self.mem_socket, channel_id, self.history_limit,
            )
            messages.extend(history)
            history_count = len(history)

        log.info("built context: %d examples + %d hard + %d history = %d total",
                 len(self.skill_examples), hard_count, history_count, len(messages))
        return messages


# ---------------------------------------------------------------------------
# Connection handler
# ---------------------------------------------------------------------------


async def handle_connection(
    service: CtxService,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
):
    try:
        msg = await recv_msg(reader)
        msg_type = msg.get("type")

        log.info("request type=%s", msg_type)

        if msg_type == "ping":
            await send_msg(writer, {"type": "pong"})

        elif msg_type == "build":
            channel_id = msg.get("channel_id")
            user_message = msg.get("user_message", "")
            log.info("build: channel=%s msg_len=%d", channel_id, len(user_message))
            context = await service.build_context(channel_id, user_message)
            await send_msg(writer, {"type": "context", "messages": context})

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

    service = CtxService(config)

    socket_path = config.get("ctx", {}).get("socket", "/run/zino/ctx.sock")
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
    parser = argparse.ArgumentParser(description="zino-ctx: context history service")
    parser.add_argument("--config", "-c",
                        default=os.environ.get("ZINO_CONFIG", "ZINO.toml"))
    args = parser.parse_args()
    asyncio.run(main(args.config))
