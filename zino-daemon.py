#!/usr/bin/env python3
"""
zino-daemon — main entrypoint, coordinator, and agentic runtime.

Responsibilities:
  - Receives (message, channel_id) from clients over UDS.
  - Assembles the full messages array:
      - System prompt from zino-ctx (get).
      - Context history from zino-ctx (build): examples, hard memories, chat history.
      - User message.
  - Runs the agentic loop in-process: rtr → parse interrupts → execute via
    zino-exc → inject results → rtr → ... until no interrupts or max_iterations.
  - Falls back to direct zino-rtr dispatch when the agentic runtime has
    nothing to offer (no tools/skills loaded).
  - Stores chat history via zino-ctx (append_history).
  - Returns the model response as a stream of packets.

All responses are streaming.  The daemon forwards chunk, tool_start,
tool_done, done, and error packets to the client in real-time.

All service connections are attempted live per-request.  If a service is
unreachable the daemon degrades gracefully — no startup probe cache that
can go stale.

UDS socket: /run/zino/daemon.sock (configurable)

Inbound message from client:
  {"message": str, "channel_id": str (optional)}

Outbound to client (always streaming):
  N × {"type": "chunk",      "delta": str}
  N × {"type": "tool_start", "kind": str, "name": str, "description": str}
  N × {"type": "tool_done",  "kind": str, "name": str}
  1 × {"type": "done",       "full_text": str}
  Error:  1 × {"type": "error", "message": str}
"""

import asyncio
import json
import os
import re
import sys
import tomllib
from pathlib import Path

from zino_utils.zino_common import send_msg, recv_msg, open_uds, setup_logging

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[daemon] Config not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Skill and tool loading (for the agentic runtime)
# ---------------------------------------------------------------------------


def load_skills(config: dict) -> dict[str, dict]:
    skills_dir = Path(config.get("skills", {}).get("dir", "skills"))
    if not skills_dir.exists():
        return {}
    skills = {}
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


def load_tool_map(config: dict) -> dict[str, str]:
    """Map function names (e.g. 'execute_bash') to executor names ('bash')."""
    tools_dir = Path(config.get("tools", {}).get("dir", "tools"))
    fn_map = {}
    if not tools_dir.exists():
        return fn_map
    for tool_file in sorted(tools_dir.glob("*.json")):
        with open(tool_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        executor = data.get("meta", {}).get("executor", tool_file.stem)
        fn_name = f"execute_{executor}"
        fn_map[fn_name] = executor
    return fn_map


# ---------------------------------------------------------------------------
# Interrupt token parsing
# ---------------------------------------------------------------------------

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)

TASK_RE = re.compile(
    r"<task>\s*(.*?)\s*</task>",
    re.DOTALL,
)


def parse_interrupts(text: str) -> list[dict]:
    """
    Parse tool_call and task blocks from LLM output.
    Returns a list of interrupt descriptors sorted by position.
    """
    interrupts = []

    for m in TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(m.group(1))
            interrupts.append({
                "kind":  "tool_call",
                "start": m.start(),
                "end":   m.end(),
                "name":  payload.get("name", ""),
                "code":  payload.get("arguments", {}).get("code", ""),
            })
        except json.JSONDecodeError as e:
            interrupts.append({
                "kind":  "tool_call",
                "start": m.start(),
                "end":   m.end(),
                "error": f"Malformed tool_call JSON: {e}",
            })

    for m in TASK_RE.finditer(text):
        try:
            payload = json.loads(m.group(1))
            interrupts.append({
                "kind":    "task",
                "start":   m.start(),
                "end":     m.end(),
                "skill":   payload.get("skill", ""),
                "request": payload.get("request", ""),
            })
        except json.JSONDecodeError as e:
            interrupts.append({
                "kind":  "task",
                "start": m.start(),
                "end":   m.end(),
                "error": f"Malformed task JSON: {e}",
            })

    interrupts.sort(key=lambda x: x["start"])
    return interrupts


# ---------------------------------------------------------------------------
# StreamParser — real-time interrupt detection on token stream
# ---------------------------------------------------------------------------

_OPEN_TAGS = {"<tool_call>": "tool_call", "<task>": "task"}
_CLOSE_TAGS = {"tool_call": "</tool_call>", "task": "</task>"}
_MAX_HOLDBACK = max(len(t) for t in _OPEN_TAGS)

_TAG_PREFIXES = set()
for _tag in _OPEN_TAGS:
    for _i in range(1, len(_tag) + 1):
        _TAG_PREFIXES.add(_tag[:_i])


class StreamParser:
    """
    Character-by-character parser that detects <tool_call>...</tool_call>
    and <task>...</task> blocks in a token stream.

    States:
      FORWARDING — normal text passes through.
      HOLDBACK   — buffering chars that might form an opening tag.
      CAPTURING  — inside a matched tag, accumulating until close tag.

    Events:
        ("text", str)      — safe text to forward to client
        ("interrupt", dict) — complete block ready for execution
    """

    def __init__(self):
        self._state = "FORWARDING"
        self._holdback = ""
        self._capture_tag = ""
        self._capture_buf = ""

    def feed(self, text: str) -> list[tuple]:
        """Feed a chunk of text, return list of events."""
        events = []
        text_accum = ""

        for ch in text:
            if self._state == "FORWARDING":
                if ch == "<":
                    if text_accum:
                        events.append(("text", text_accum))
                        text_accum = ""
                    self._holdback = ch
                    self._state = "HOLDBACK"
                else:
                    text_accum += ch

            elif self._state == "HOLDBACK":
                self._holdback += ch
                if self._holdback in _TAG_PREFIXES:
                    if self._holdback in _OPEN_TAGS:
                        self._capture_tag = _OPEN_TAGS[self._holdback]
                        self._capture_buf = ""
                        self._state = "CAPTURING"
                else:
                    text_accum += self._holdback
                    self._holdback = ""
                    self._state = "FORWARDING"

            elif self._state == "CAPTURING":
                self._capture_buf += ch
                close_tag = _CLOSE_TAGS[self._capture_tag]
                if self._capture_buf.endswith(close_tag):
                    inner = self._capture_buf[:-len(close_tag)]
                    event = self._parse_interrupt(self._capture_tag, inner)
                    events.append(("interrupt", event))
                    self._capture_tag = ""
                    self._capture_buf = ""
                    self._state = "FORWARDING"

        if text_accum:
            events.append(("text", text_accum))

        return events

    def flush(self) -> list[tuple]:
        """End of stream — flush any remaining buffered content as text."""
        events = []
        if self._state == "HOLDBACK" and self._holdback:
            events.append(("text", self._holdback))
            self._holdback = ""
        elif self._state == "CAPTURING" and self._capture_buf:
            open_tag = f"<{self._capture_tag}>"
            events.append(("text", open_tag + self._capture_buf))
            self._capture_buf = ""
            self._capture_tag = ""
        self._state = "FORWARDING"
        return events

    @staticmethod
    def _parse_interrupt(kind: str, inner: str) -> dict:
        """Parse the JSON content inside a tag block."""
        raw = inner.strip()
        result = {"kind": kind, "raw": raw}
        try:
            payload = json.loads(raw)
            if kind == "tool_call":
                result["name"] = payload.get("name", "")
                result["code"] = payload.get("arguments", {}).get("code", "")
            elif kind == "task":
                result["skill"] = payload.get("skill", "")
                result["request"] = payload.get("request", "")
        except json.JSONDecodeError as e:
            result["error"] = f"Malformed {kind} JSON: {e}"
        return result

    def raw_for_interrupt(self, interrupt: dict) -> str:
        """Reconstruct the raw XML block from an interrupt event."""
        kind = interrupt["kind"]
        raw = interrupt.get("raw", "")
        return f"<{kind}>{raw}</{kind}>"


# ---------------------------------------------------------------------------
# RTR interaction
# ---------------------------------------------------------------------------


async def rtr_infer(
    rtr_socket: str, messages: list[dict], temperature: float, top_p: float,
) -> str:
    """Send messages to zino-rtr for inference (non-streaming)."""
    reader, writer = await open_uds(rtr_socket)
    try:
        await send_msg(writer, {
            "type":        "infer",
            "messages":    messages,
            "temperature": temperature,
            "top_p":       top_p,
            "stream":      False,
        })
        packet = await recv_msg(reader)
        if packet.get("type") == "error":
            raise RuntimeError(f"rtr error: {packet.get('message')}")
        return packet.get("content", "")
    finally:
        writer.close()
        await writer.wait_closed()


async def rtr_infer_stream(
    rtr_socket: str, messages: list[dict], temperature: float, top_p: float,
):
    """
    Async generator: stream inference from zino-rtr.
    Yields ("chunk", delta_str) and finally ("done", full_text_str).
    """
    reader, writer = await open_uds(rtr_socket)
    try:
        await send_msg(writer, {
            "type":        "infer",
            "messages":    messages,
            "temperature": temperature,
            "top_p":       top_p,
            "stream":      True,
        })
        while True:
            packet = await recv_msg(reader)
            ptype = packet.get("type")
            if ptype == "chunk":
                yield ("chunk", packet.get("delta", ""))
            elif ptype == "done":
                yield ("done", packet.get("full_text", ""))
                break
            elif ptype == "error":
                raise RuntimeError(f"rtr stream error: {packet.get('message')}")
            else:
                break
    finally:
        writer.close()
        await writer.wait_closed()


async def probe_rtr(rtr_socket: str) -> dict:
    """Ask zino-rtr for its capabilities.  Returns {} on failure."""
    try:
        reader, writer = await open_uds(rtr_socket)
        await send_msg(writer, {"type": "capabilities"})
        caps = await recv_msg(reader)
        writer.close()
        await writer.wait_closed()
        return caps
    except Exception as e:
        log.warning("could not probe rtr at %s: %s", rtr_socket, e)
        return {}


# ---------------------------------------------------------------------------
# EXC interaction
# ---------------------------------------------------------------------------


async def exc_execute(exc_socket: str, executor: str, code: str) -> dict:
    """Send code to zino-exc for execution."""
    reader, writer = await open_uds(exc_socket)
    try:
        await send_msg(writer, {
            "type":     "execute",
            "executor": executor,
            "code":     code,
        })
        packet = await recv_msg(reader)
        if packet.get("type") == "error":
            return {"exit_code": -1, "stdout": "",
                    "stderr": packet.get("message", "Unknown error")}
        return {
            "exit_code": packet.get("exit_code", -1),
            "stdout":    packet.get("stdout", ""),
            "stderr":    packet.get("stderr", ""),
        }
    finally:
        writer.close()
        await writer.wait_closed()


# ---------------------------------------------------------------------------
# CTX interaction (system prompt, context assembly, history storage)
# ---------------------------------------------------------------------------


async def fetch_system_prompt(ctx_socket: str) -> str | None:
    """
    Request the current built system prompt from zino-ctx.
    Returns None if zino-ctx is unavailable — daemon continues in bare mode.
    """
    try:
        reader, writer = await open_uds(ctx_socket)
        await send_msg(writer, {"type": "get"})
        msg = await recv_msg(reader)
        writer.close()
        await writer.wait_closed()
        if msg.get("type") == "system_prompt":
            content = msg.get("content", "")
            return content if content else None
        return None
    except Exception as e:
        log.warning("could not reach zino-ctx for system prompt: %s — bare mode.", e)
        return None


async def fetch_context(
    ctx_socket: str, channel_id: str | None, user_message: str,
) -> list[dict]:
    """
    Request context history from zino-ctx.
    Returns [] if zino-ctx is unavailable.
    """
    try:
        reader, writer = await open_uds(ctx_socket)
        await send_msg(writer, {
            "type":         "build",
            "channel_id":   channel_id,
            "user_message": user_message,
        })
        msg = await recv_msg(reader)
        writer.close()
        await writer.wait_closed()
        if msg.get("type") == "context":
            return msg.get("messages", [])
        return []
    except Exception as e:
        log.warning("could not reach zino-ctx for context: %s — no context.", e)
        return []


async def store_history(
    ctx_socket: str, channel_id: str, user_msg: str, assistant_msg: str,
):
    """Append a user+assistant exchange to channel history via zino-ctx."""
    try:
        reader, writer = await open_uds(ctx_socket)
        await send_msg(writer, {
            "type":       "append_history",
            "channel_id": channel_id,
            "messages": [
                {"role": "user",      "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
        })
        await recv_msg(reader)
        writer.close()
        await writer.wait_closed()
    except Exception as e:
        log.warning("could not store history in zino-ctx: %s", e)


# ---------------------------------------------------------------------------
# Skill pipeline
# ---------------------------------------------------------------------------


def format_exec_result(result: dict) -> str:
    """Format an execution result into a human-readable string."""
    parts = []
    if result["stdout"]:
        parts.append(result["stdout"])
    if result["stderr"]:
        parts.append(f"[stderr]\n{result['stderr']}")
    if result["exit_code"] != 0:
        parts.append(f"[exit code: {result['exit_code']}]")
    return "\n".join(parts) if parts else "(no output)"


async def fetch_skill_context(ctx_socket: str, skill_name: str) -> list[dict]:
    """Fetch skill examples from the _skill_{name} context channel."""
    try:
        reader, writer = await open_uds(ctx_socket)
        await send_msg(writer, {
            "type":       "get_history",
            "channel_id": f"_skill_{skill_name}",
        })
        msg = await recv_msg(reader)
        writer.close()
        await writer.wait_closed()
        if msg.get("type") == "history":
            return msg.get("messages", [])
        return []
    except Exception as e:
        log.warning("could not fetch skill context for %s: %s", skill_name, e)
        return []


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def assemble_messages(
    user_message: str,
    system_prompt: str | None,
    context_messages: list[dict] | None = None,
) -> list[dict]:
    """
    Assemble the full messages array.

      [system]      ← from zino-ctx if available
      [context]     ← from zino-ctx: examples, hard memories, chat history
      [user]        ← always present
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if context_messages:
        messages.extend(context_messages)

    messages.append({"role": "user", "content": user_message})
    return messages


# ---------------------------------------------------------------------------
# Agentic runtime
# ---------------------------------------------------------------------------


class AgenticRuntime:
    """In-process agentic loop."""

    def __init__(self, config: dict):
        self.rtr_socket = config.get("rtr", {}).get("socket", "/run/zino/rtr.sock")
        self.exc_socket = config.get("exc", {}).get("socket", "/run/zino/exc.sock")
        self.ctx_socket = config.get("ctx", {}).get("socket", "/run/zino/ctx.sock")
        self.default_max_iter = config.get("agent", {}).get("max_iterations", 5)

        self.skills = load_skills(config)
        self.tool_map = load_tool_map(config)

        log.info("agentic runtime: skills=%s", list(self.skills.keys()))
        log.info("agentic runtime: tool_map=%s", self.tool_map)
        log.info("agentic runtime: default max_iterations=%d", self.default_max_iter)

    @property
    def available(self) -> bool:
        """Whether the agentic runtime has tools or skills to work with."""
        return bool(self.skills or self.tool_map)

    async def run_streaming(
        self,
        messages: list[dict],
        temperature: float,
        top_p: float,
        max_iterations: int,
        client_writer: asyncio.StreamWriter,
    ) -> str:
        """
        Streaming agentic loop.  Forwards text chunks to client_writer in
        real-time and executes interrupts as soon as their closing tags arrive.
        Sends {"type": "done", "full_text": ..., "iterations": N} at the end.
        Returns the full response text for history storage.
        """
        iteration = 0
        parser = StreamParser()
        full_text_all = ""

        while iteration < max_iterations:
            iteration += 1
            log.info("stream iteration %d/%d — sending %d messages to rtr",
                     iteration, max_iterations, len(messages))

            continuation_text = ""
            interrupts_found = []

            async for kind, payload in rtr_infer_stream(
                self.rtr_socket, messages, temperature, top_p,
            ):
                if kind == "chunk":
                    events = parser.feed(payload)
                    for etype, edata in events:
                        if etype == "text":
                            await send_msg(client_writer, {"type": "chunk", "delta": edata})
                            continuation_text += edata
                            log.debug("streamed text chunk: %d chars", len(edata))
                        elif etype == "interrupt":
                            interrupts_found.append(edata)
                            raw_block = parser.raw_for_interrupt(edata)
                            continuation_text += raw_block

                            if edata["kind"] == "tool_call":
                                description = edata.get("name", "unknown tool")
                            elif edata["kind"] == "task":
                                description = f'{edata.get("skill", "unknown")}: {edata.get("request", "")[:80]}'
                            else:
                                description = "working..."

                            tool_name = edata.get("name") or edata.get("skill", "")
                            await send_msg(client_writer, {
                                "type": "tool_start",
                                "kind": edata["kind"],
                                "name": tool_name,
                                "description": description,
                            })

                            result_text = await self._execute_interrupt(
                                edata, temperature, top_p,
                            )

                            await send_msg(client_writer, {
                                "type": "tool_done",
                                "kind": edata["kind"],
                                "name": tool_name,
                            })

                            tag = ("tool_response" if edata["kind"] == "tool_call"
                                   else "task_response")
                            response_block = f"\n<{tag}>\n{result_text}\n</{tag}>\n"
                            continuation_text += response_block
                            log.info("interrupt executed mid-stream: %s", edata["kind"])
                            log.debug("response block:\n%s", response_block)

                elif kind == "done":
                    pass  # full_iter_text from rtr; we track our own

            # Flush remaining holdback
            for etype, edata in parser.flush():
                if etype == "text":
                    await send_msg(client_writer, {"type": "chunk", "delta": edata})
                    continuation_text += edata

            parser = StreamParser()

            if not interrupts_found:
                log.info("no interrupts in stream iteration %d — loop complete.", iteration)
                full_text_all = continuation_text
                break

            log.info("stream iteration %d found %d interrupt(s)", iteration, len(interrupts_found))

            messages = messages + [
                {"role": "assistant", "content": continuation_text}
            ]
            full_text_all = continuation_text

        if iteration >= max_iterations:
            log.warning("max iterations (%d) reached in streaming mode.", max_iterations)

        await send_msg(client_writer, {
            "type": "done",
            "full_text": full_text_all,
            "iterations": iteration,
        })
        log.info("streaming run complete: %d iterations, %d chars",
                 iteration, len(full_text_all))
        return full_text_all

    async def _execute_interrupt(
        self, intr: dict, temperature: float, top_p: float,
    ) -> str:
        """Execute a single interrupt (tool_call or task)."""
        if "error" in intr:
            log.warning("interrupt parse error: %s", intr["error"])
            return intr["error"]

        if intr["kind"] == "tool_call":
            return await self._handle_tool_call(intr)
        elif intr["kind"] == "task":
            return await self._handle_task(intr, temperature, top_p)
        return "Error: Unknown interrupt kind."

    async def _handle_tool_call(self, intr: dict) -> str:
        """Execute a tool call via zino-exc."""
        fn_name = intr["name"]
        code = intr["code"]

        executor = self.tool_map.get(fn_name)
        if not executor:
            log.error("unknown tool function: %s", fn_name)
            return f"Error: Unknown tool function '{fn_name}'"

        if not code.strip():
            return "Error: Empty code block."

        log.info("executing tool: %s (executor: %s) code_len=%d",
                 fn_name, executor, len(code))
        log.debug("tool code:\n%s", code)
        result = await exc_execute(self.exc_socket, executor, code)
        log.info("tool result: exit_code=%d stdout_len=%d stderr_len=%d",
                 result["exit_code"], len(result["stdout"]), len(result["stderr"]))
        return format_exec_result(result)

    async def _handle_task(
        self, intr: dict, temperature: float, top_p: float,
    ) -> str:
        """Spawn a sub-agent to handle a task delegation."""
        skill_name = intr["skill"]
        request = intr["request"]

        skill = self.skills.get(skill_name)
        if not skill:
            log.error("unknown skill: %s", skill_name)
            return f"Error: Unknown skill '{skill_name}'"
        if not request.strip():
            return "Error: Empty task request."

        log.info("spawning sub-agent for skill: %s request_len=%d",
                 skill_name, len(request))
        log.debug("skill request:\n%s", request)

        try:
            # Build allowed tool map: only tools listed in the skill definition
            allowed_tool_map = {}
            for tool_executor in skill.get("tools", []):
                fn_name = f"execute_{tool_executor}"
                if fn_name in self.tool_map:
                    allowed_tool_map[fn_name] = self.tool_map[fn_name]
            log.debug("sub-agent allowed tools: %s", allowed_tool_map)

            # Fetch skill examples from ctx channel
            examples = await fetch_skill_context(self.ctx_socket, skill_name)
            log.debug("sub-agent loaded %d example messages", len(examples))

            # Assemble sub-agent messages
            messages: list[dict] = []
            system_prompt = skill.get("system", "")
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(examples)
            messages.append({"role": "user", "content": request})

            return await self._run_subagent(
                messages, allowed_tool_map, temperature, top_p,
                self.default_max_iter,
            )
        except Exception as e:
            log.error("sub-agent error: %s", e)
            return f"Sub-agent error: {e}"

    async def _run_subagent(
        self,
        messages: list[dict],
        allowed_tool_map: dict[str, str],
        temperature: float,
        top_p: float,
        max_iterations: int,
    ) -> str:
        """
        Simplified, non-streaming agentic loop for sub-agents.
        Only parses tool_call interrupts (no task nesting).
        Returns the final response text.
        """
        iteration = 0
        last_response = ""

        while iteration < max_iterations:
            iteration += 1
            log.info("sub-agent iteration %d/%d — %d messages",
                     iteration, max_iterations, len(messages))

            response = await rtr_infer(
                self.rtr_socket, messages, temperature, top_p,
            )
            log.debug("sub-agent response (%d chars):\n%s", len(response), response)

            # Parse only tool_call interrupts (no task — sub-agents can't nest)
            tool_calls = list(TOOL_CALL_RE.finditer(response))
            if not tool_calls:
                log.info("sub-agent iteration %d — no tool calls, done.", iteration)
                last_response = response
                break

            log.info("sub-agent iteration %d found %d tool call(s)",
                     iteration, len(tool_calls))

            # Build continuation with tool responses injected
            continuation = response
            for m in tool_calls:
                try:
                    payload = json.loads(m.group(1))
                    fn_name = payload.get("name", "")
                    code = payload.get("arguments", {}).get("code", "")

                    executor = allowed_tool_map.get(fn_name)
                    if not executor:
                        result_text = f"Error: Tool '{fn_name}' not allowed for this sub-agent."
                        log.warning("sub-agent tried disallowed tool: %s", fn_name)
                    elif not code.strip():
                        result_text = "Error: Empty code block."
                    else:
                        log.info("sub-agent executing: %s (executor: %s)",
                                 fn_name, executor)
                        result = await exc_execute(self.exc_socket, executor, code)
                        result_text = format_exec_result(result)
                        log.info("sub-agent exec result: exit_code=%d",
                                 result["exit_code"])

                except json.JSONDecodeError as e:
                    result_text = f"Malformed tool_call JSON: {e}"

                continuation += f"\n<tool_response>\n{result_text}\n</tool_response>\n"

            messages = messages + [{"role": "assistant", "content": continuation}]
            last_response = continuation

        if iteration >= max_iterations:
            log.warning("sub-agent reached max iterations (%d).", max_iterations)

        return last_response


# ---------------------------------------------------------------------------
# RTR dispatch (fallback when agentic runtime is not applicable)
# ---------------------------------------------------------------------------


async def dispatch_to_rtr(
    rtr_socket: str,
    messages: list[dict],
    temperature: float,
    top_p: float,
    rtr_caps: dict,
    client_writer: asyncio.StreamWriter,
) -> str | None:
    """
    Send assembled messages to zino-rtr (fallback when agentic loop is not needed).
    Always streams.  Forward chunk/done/error packets to client.
    Returns the full response text (for history storage), or None on error.
    """
    do_stream = rtr_caps.get("streaming", False)

    try:
        reader, writer = await open_uds(rtr_socket)
    except Exception as e:
        await send_msg(client_writer,
                       {"type": "error",
                        "message": f"Could not connect to rtr: {e}"})
        return None

    try:
        await send_msg(writer, {
            "type":        "infer",
            "messages":    messages,
            "temperature": temperature,
            "top_p":       top_p,
            "stream":      do_stream,
        })

        if do_stream:
            full_text = None
            while True:
                packet = await recv_msg(reader)
                await send_msg(client_writer, packet)
                if packet.get("type") == "done":
                    full_text = packet.get("full_text", "")
                    break
                if packet.get("type") == "error":
                    break
            return full_text
        else:
            packet = await recv_msg(reader)
            if packet.get("type") == "response":
                content = packet.get("content", "")
                await send_msg(client_writer, {"type": "chunk", "delta": content})
                await send_msg(client_writer, {"type": "done", "full_text": content})
                return content
            await send_msg(client_writer, packet)
            return None

    except Exception as e:
        await send_msg(client_writer,
                       {"type": "error",
                        "message": f"rtr dispatch error: {e}"})
        return None
    finally:
        writer.close()
        await writer.wait_closed()


# ---------------------------------------------------------------------------
# Connection handler
# ---------------------------------------------------------------------------


async def handle_client(
    config: dict,
    sockets: dict,
    rtr_caps_cache: dict,
    runtime: AgenticRuntime,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
):
    peer = writer.get_extra_info("peername", "<unknown>")
    try:
        msg = await recv_msg(reader)

        user_message = msg.get("message", "").strip()
        if not user_message:
            await send_msg(writer, {"type": "error", "message": "Empty message."})
            return

        channel_id = msg.get("channel_id")

        log.info("request: msg_len=%d channel=%s",
                 len(user_message), channel_id)
        log.debug("user message: %s", user_message)

        agent       = config.get("agent", {})
        temperature = float(agent.get("temperature", 0.7))
        top_p       = float(agent.get("top_p", 1.0))
        max_iter    = int(agent.get("max_iterations", 5))

        # Lazy re-probe rtr capabilities if the cache is empty
        if not rtr_caps_cache:
            fresh = await probe_rtr(sockets["rtr"])
            if fresh:
                rtr_caps_cache.update(fresh)
                log.info("rtr capabilities acquired: %s", rtr_caps_cache)

        # 1. System prompt from zino-ctx (optional — tries every request)
        system_prompt = await fetch_system_prompt(sockets["ctx"])
        log.info("system prompt: %s",
                 f"{len(system_prompt)} chars" if system_prompt else "none (bare mode)")

        # 2. Context from zino-ctx (optional — tries every request)
        context_messages = await fetch_context(
            sockets["ctx"], channel_id, user_message,
        )
        log.info("context: %d messages", len(context_messages))

        # 3. Assemble messages
        messages = assemble_messages(user_message, system_prompt, context_messages)
        log.info("assembled %d messages for inference", len(messages))

        # 4. Route through agentic runtime or fall back to direct rtr
        response_content = None
        if runtime.available:
            try:
                response_content = await runtime.run_streaming(
                    messages, temperature, top_p, max_iter, writer,
                )
            except Exception as e:
                log.error("agentic runtime error: %s", e)
                await send_msg(writer, {
                    "type": "error",
                    "message": f"Agentic runtime error: {e}",
                })
        else:
            log.info("no tools/skills loaded — using direct rtr dispatch")
            response_content = await dispatch_to_rtr(
                sockets["rtr"], messages, temperature, top_p,
                rtr_caps_cache, writer,
            )

        # 5. Store history in zino-ctx (optional — tries every request)
        if channel_id and response_content:
            await store_history(
                sockets["ctx"], channel_id, user_message, response_content,
            )
            log.info("history stored for channel=%s", channel_id)

    except asyncio.IncompleteReadError:
        pass
    except Exception as e:
        log.error("handler error (%s): %s", peer, e)
        try:
            await send_msg(writer, {"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        writer.close()
        await writer.wait_closed()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(config_path: str):
    config = load_config(config_path)

    global log
    log = setup_logging("zino.daemon", config)

    sockets = {
        "daemon": config.get("daemon", {}).get("socket", "/run/zino/daemon.sock"),
        "rtr":    config.get("rtr", {}).get("socket", "/run/zino/rtr.sock"),
        "exc":    config.get("exc", {}).get("socket", "/run/zino/exc.sock"),
        "ctx":    config.get("ctx", {}).get("socket", "/run/zino/ctx.sock"),
    }

    runtime = AgenticRuntime(config)

    # Initial rtr probe — mutable dict so handle_client can update it lazily
    rtr_caps_cache = await probe_rtr(sockets["rtr"])
    if rtr_caps_cache:
        log.info("rtr capabilities: %s", rtr_caps_cache)
    else:
        log.info("rtr not yet reachable — will probe on first request.")

    socket_path = sockets["daemon"]
    Path(socket_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        Path(socket_path).unlink()
    except FileNotFoundError:
        pass

    server = await asyncio.start_unix_server(
        lambda r, w: handle_client(config, sockets, rtr_caps_cache, runtime, r, w),
        path=socket_path,
    )

    log.info("listening on %s", socket_path)
    async with server:
        await server.serve_forever()


log = None  # set in main()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="zino-daemon: main coordinator and agentic runtime")
    parser.add_argument("--config", "-c",
                        default=os.environ.get("ZINO_CONFIG", "ZINO.toml"))
    args = parser.parse_args()
    asyncio.run(main(args.config))
