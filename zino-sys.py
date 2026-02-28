#!/usr/bin/env python3
"""
zino-sys — system prompt service.

Responsibilities:
  - Loads SYSTEM.md template from the working directory.
  - Loads tool and skill definitions from their configured directories.
  - Builds the system prompt by replacing template placeholders.
  - Serves the built prompt to zino-daemon on request.
  - Rebuilds when told to (tool/skill reload, soft memory update).

UDS socket: /run/zino/sys.sock (configurable)

Inbound message types:
  {"type": "get"}
      → {"type": "system_prompt", "content": str}

  {"type": "reload"}
      Reloads tools, skills, re-renders the template.
      → {"type": "system_prompt", "content": str}

  {"type": "set_soft_memory", "content": str}
      Replaces the current soft memory block and re-renders.
      → {"type": "system_prompt", "content": str}

  {"type": "ping"}
      → {"type": "pong"}

Placeholders in SYSTEM.md:
  %%PERSONALITY%%       — from [sys] personality in config, or empty
  %%TOOL_LIST%%         — built from tools/ directory
  %%SKILL_LINES%%       — built from skills/ directory
  %%MAX_ITERATIONS%%    — from [agent] max_iterations in config
  %%SOFT_MEMORIES%%     — runtime-mutable; starts empty or from config
"""

import asyncio
import json
import os
import sys
import tomllib
from pathlib import Path

from zino_common import send_msg, recv_msg, setup_logging

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[sys] Config not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Tool and skill loading  (ported from KINO4.py)
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
            "meta":        data.get("meta", {}),
            "interpreter": data.get("interpreter", []),
            "fabricator":  data.get("fabricator", []),
        }
    return skills


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------


def load_template(config: dict) -> str:
    path = config.get("sys", {}).get("template", "SYSTEM.md")
    p = Path(path)
    if not p.exists():
        log.warning("SYSTEM.md not found at %s — serving empty system prompt.", path)
        return ""
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _tool_fn_name(executor: str) -> str:
    return f"execute_{executor}"


def build_prompt(config: dict, tools: dict, skills: dict, soft_memory: str) -> str:
    template = load_template(config)
    if not template:
        return ""

    # %%PERSONALITY%%
    personality = config.get("sys", {}).get("personality", "")

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


# ---------------------------------------------------------------------------
# Service state
# ---------------------------------------------------------------------------


class SysService:
    def __init__(self, config: dict):
        self.config      = config
        self.soft_memory = config.get("sys", {}).get("soft_memory", "")
        self.tools       = load_tools(config)
        self.skills      = load_skills(config)
        self._prompt     = build_prompt(config, self.tools, self.skills, self.soft_memory)
        log.info("loaded %d tool(s), %d skill(s).", len(self.tools), len(self.skills))

    @property
    def prompt(self) -> str:
        return self._prompt

    def reload(self) -> str:
        self.tools   = load_tools(self.config)
        self.skills  = load_skills(self.config)
        self._prompt = build_prompt(self.config, self.tools, self.skills, self.soft_memory)
        log.info("reloaded: %d tool(s), %d skill(s).", len(self.tools), len(self.skills))
        return self._prompt

    def set_soft_memory(self, content: str) -> str:
        self.soft_memory = content
        self._prompt     = build_prompt(self.config, self.tools, self.skills, self.soft_memory)
        log.info("soft memory updated, prompt rebuilt.")
        return self._prompt


# ---------------------------------------------------------------------------
# Connection handler
# ---------------------------------------------------------------------------


async def handle_connection(service: SysService, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        msg = await recv_msg(reader)
        msg_type = msg.get("type")

        log.info("request type=%s", msg_type)

        if msg_type == "ping":
            await send_msg(writer, {"type": "pong"})

        elif msg_type == "get":
            await send_msg(writer, {"type": "system_prompt", "content": service.prompt})
            log.debug("served system prompt (%d chars)", len(service.prompt))

        elif msg_type == "reload":
            prompt = service.reload()
            await send_msg(writer, {"type": "system_prompt", "content": prompt})

        elif msg_type == "set_soft_memory":
            content = msg.get("content", "")
            prompt  = service.set_soft_memory(content)
            await send_msg(writer, {"type": "system_prompt", "content": prompt})

        else:
            await send_msg(writer, {"type": "error", "message": f"Unknown message type: {msg_type}"})
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
    config  = load_config(config_path)

    global log
    log = setup_logging("zino.sys", config)

    service = SysService(config)

    socket_path = config.get("sys", {}).get("socket", "/run/zino/sys.sock")
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
    parser = argparse.ArgumentParser(description="zino-sys: system prompt service")
    parser.add_argument("--config", "-c", default=os.environ.get("ZINO_CONFIG", "ZINO.toml"))
    args = parser.parse_args()
    asyncio.run(main(args.config))
