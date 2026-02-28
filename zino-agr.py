#!/usr/bin/env python3
"""
zino-agr — agentic runtime service.

Responsibilities:
  - Manages the inference loop: rtr → parse → execute → rtr → ...
  - Parses LLM output for <tool_call> and <task> interrupt tokens.
  - Coordinates with zino-exc for code execution.
  - Runs skill pipelines (interpreter → fabricator → execute) for <task> blocks.
  - Enforces max_iterations.

UDS socket: /run/zino/agr.sock (configurable)

Inbound message types:
  {"type": "run", "messages": [...], "temperature": float, "top_p": float,
   "max_iterations": int (optional)}
      → N × {"type": "chunk",      "delta": str}
        N × {"type": "tool_start", "kind": str, "name": str, "description": str}
        N × {"type": "tool_done",  "kind": str, "name": str}
        1 × {"type": "done",       "full_text": str, "iterations": int}
      All responses are streaming.

  {"type": "ping"}
      → {"type": "pong"}

The messages array should be the fully assembled prompt
(system + context + user).  agr handles the inference loop internally,
calling zino-rtr and zino-exc as needed.
"""

import asyncio
import json
import os
import re
import sys
import tomllib
from pathlib import Path

from zino_common import send_msg, recv_msg, open_connection, start_server, setup_logging

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[agr] Config not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Skill loading
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
            "meta":        data.get("meta", {}),
            "interpreter": data.get("interpreter", []),
            "fabricator":  data.get("fabricator", []),
        }
    return skills


# ---------------------------------------------------------------------------
# Tool name → executor mapping
# ---------------------------------------------------------------------------


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

# Tag definitions for the parser
_OPEN_TAGS = {"<tool_call>": "tool_call", "<task>": "task"}
_CLOSE_TAGS = {"tool_call": "</tool_call>", "task": "</task>"}
_MAX_HOLDBACK = max(len(t) for t in _OPEN_TAGS)  # 11 for "<tool_call>"

# All possible open tag prefixes for holdback matching
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

    Usage:
        parser = StreamParser()
        events = parser.feed("some text <tool_call>{...}")
        events = parser.feed("...</tool_call> more")
        events = parser.flush()  # end-of-stream

    Events:
        ("text", str)      — safe text to forward to client
        ("interrupt", dict) — complete block ready for execution
    """

    def __init__(self):
        self._state = "FORWARDING"  # FORWARDING | HOLDBACK | CAPTURING
        self._holdback = ""         # buffer during HOLDBACK
        self._capture_tag = ""      # which tag we're capturing ("tool_call" or "task")
        self._capture_buf = ""      # accumulated content inside tag (includes open tag)

    def feed(self, text: str) -> list[tuple]:
        """Feed a chunk of text, return list of events."""
        events = []
        text_accum = ""

        for ch in text:
            if self._state == "FORWARDING":
                if ch == "<":
                    # Potential tag start — flush accumulated text
                    if text_accum:
                        events.append(("text", text_accum))
                        text_accum = ""
                    self._holdback = ch
                    self._state = "HOLDBACK"
                else:
                    text_accum += ch

            elif self._state == "HOLDBACK":
                self._holdback += ch
                # Check if holdback still matches a prefix of any open tag
                if self._holdback in _TAG_PREFIXES:
                    # Check if we have a complete open tag
                    if self._holdback in _OPEN_TAGS:
                        self._capture_tag = _OPEN_TAGS[self._holdback]
                        self._capture_buf = ""
                        self._state = "CAPTURING"
                    # else keep holding back
                else:
                    # No tag can match — flush holdback as text
                    text_accum += self._holdback
                    self._holdback = ""
                    self._state = "FORWARDING"

            elif self._state == "CAPTURING":
                self._capture_buf += ch
                close_tag = _CLOSE_TAGS[self._capture_tag]
                if self._capture_buf.endswith(close_tag):
                    # Complete block found — parse the inner content
                    inner = self._capture_buf[:-len(close_tag)]
                    event = self._parse_interrupt(self._capture_tag, inner)
                    events.append(("interrupt", event))
                    self._capture_tag = ""
                    self._capture_buf = ""
                    self._state = "FORWARDING"

        # Flush any remaining accumulated text
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
            # Unclosed tag — flush as text (include the open tag)
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
    reader, writer = await open_connection(rtr_socket)
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
    reader, writer = await open_connection(rtr_socket)
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


# ---------------------------------------------------------------------------
# EXC interaction
# ---------------------------------------------------------------------------


async def exc_execute(exc_socket: str, executor: str, code: str) -> dict:
    """Send code to zino-exc for execution."""
    reader, writer = await open_connection(exc_socket)
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


async def run_skill_pipeline(
    skill: dict,
    request: str,
    rtr_socket: str,
    exc_socket: str,
    temperature: float,
    top_p: float,
) -> str:
    """
    Run the KBI skill pipeline:
      1. Interpreter: understand the request and produce code
      2. Fabricator:  clean up the code into an executable script
      3. Execute:     run the script via zino-exc
    """
    meta = skill.get("meta", {})
    executor = meta.get("executor", "bash")

    # --- Phase 1: Interpreter ---
    interp_messages = list(skill.get("interpreter", []))
    interp_messages.append({"role": "user", "content": request})

    interp_response = await rtr_infer(
        rtr_socket, interp_messages, temperature, top_p,
    )
    log.info("skill interpreter responded: %d chars", len(interp_response))
    log.debug("interpreter response:\n%s", interp_response)

    # --- Phase 2: Fabricator ---
    fab_messages = list(skill.get("fabricator", []))
    kcr_prompt = f"{request} [KCR] {interp_response}"
    fab_messages.append({"role": "user", "content": kcr_prompt})

    fab_response = await rtr_infer(
        rtr_socket, fab_messages, temperature, top_p,
    )
    log.info("skill fabricator responded: %d chars", len(fab_response))
    log.debug("fabricator response:\n%s", fab_response)

    # --- Phase 3: Execute ---
    result = await exc_execute(exc_socket, executor, fab_response)
    log.info("skill execution: exit_code=%d", result["exit_code"])
    return format_exec_result(result)


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------


class AgrService:
    def __init__(self, config: dict):
        self.config = config
        agr_cfg = config.get("agr", {})
        self.rtr_socket = config.get("rtr", {}).get("socket", "/run/zino/rtr.sock")
        self.exc_socket = config.get("exc", {}).get("socket", "/run/zino/exc.sock")
        self.default_max_iter = agr_cfg.get("max_iterations", 5)

        self.skills = load_skills(config)
        self.tool_map = load_tool_map(config)

        log.info("skills loaded: %s", list(self.skills.keys()))
        log.info("tool map: %s", self.tool_map)
        log.info("default max iterations: %d", self.default_max_iter)

    async def run(
        self,
        messages: list[dict],
        temperature: float,
        top_p: float,
        max_iterations: int,
    ) -> dict:
        """
        Run the agentic loop.
        Returns {"content": str, "iterations": int}.
        """
        iteration = 0
        full_response = ""

        while iteration < max_iterations:
            iteration += 1
            log.info("iteration %d/%d — sending %d messages to rtr",
                     iteration, max_iterations, len(messages))

            response = await rtr_infer(
                self.rtr_socket, messages, temperature, top_p,
            )

            log.info("rtr response: %d chars", len(response))
            log.debug("rtr response text:\n%s", response)

            interrupts = parse_interrupts(response)

            if not interrupts:
                log.info("no interrupts found — loop complete.")
                full_response = response
                break

            log.info("found %d interrupt(s): %s", len(interrupts),
                     [(i["kind"], i.get("name") or i.get("skill", "")) for i in interrupts])

            # Process interrupts, injecting response blocks after each one.
            # Track offset as each insertion shifts subsequent positions.
            continued_text = response
            offset = 0

            for intr in interrupts:
                if "error" in intr:
                    tag = ("tool_response" if intr["kind"] == "tool_call"
                           else "task_response")
                    result_text = intr["error"]
                    log.warning("interrupt parse error: %s", result_text)
                else:
                    if intr["kind"] == "tool_call":
                        result_text = await self._handle_tool_call(intr)
                        tag = "tool_response"
                    elif intr["kind"] == "task":
                        result_text = await self._handle_task(
                            intr, temperature, top_p,
                        )
                        tag = "task_response"
                    else:
                        continue

                log.debug("<%s> result:\n%s", tag, result_text)

                response_block = f"\n<{tag}>\n{result_text}\n</{tag}>\n"
                insert_pos = intr["end"] + offset
                continued_text = (
                    continued_text[:insert_pos]
                    + response_block
                    + continued_text[insert_pos:]
                )
                offset += len(response_block)

            # Append the assistant message with call+response for continuation
            messages = messages + [
                {"role": "assistant", "content": continued_text}
            ]
            full_response = continued_text

        if iteration >= max_iterations:
            log.warning("max iterations (%d) reached.", max_iterations)

        return {"content": full_response, "iterations": iteration}

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
        """Run a skill pipeline for a task delegation."""
        skill_name = intr["skill"]
        request = intr["request"]

        skill = self.skills.get(skill_name)
        if not skill:
            log.error("unknown skill: %s", skill_name)
            return f"Error: Unknown skill '{skill_name}'"
        if not request.strip():
            return "Error: Empty task request."

        log.info("running skill pipeline: %s request_len=%d",
                 skill_name, len(request))
        log.debug("skill request:\n%s", request)
        try:
            return await run_skill_pipeline(
                skill, request,
                self.rtr_socket, self.exc_socket,
                temperature, top_p,
            )
        except Exception as e:
            log.error("skill pipeline error: %s", e)
            return f"Skill pipeline error: {e}"

    async def run_streaming(
        self,
        messages: list[dict],
        temperature: float,
        top_p: float,
        max_iterations: int,
        client_writer,
    ):
        """
        Streaming agentic loop.  Forwards text chunks to client_writer in
        real-time and executes interrupts as soon as their closing tags arrive.
        Sends {"type": "done", "full_text": ..., "iterations": N} at the end.
        """
        iteration = 0
        parser = StreamParser()
        full_text_all = ""

        while iteration < max_iterations:
            iteration += 1
            log.info("stream iteration %d/%d — sending %d messages to rtr",
                     iteration, max_iterations, len(messages))

            # Track everything produced in this iteration for continuation
            continuation_text = ""
            interrupts_found = []
            full_iter_text = ""

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
                            # Add the raw XML block to continuation
                            raw_block = parser.raw_for_interrupt(edata)
                            continuation_text += raw_block

                            # Build description and notify client
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

                            # Execute immediately
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
                    full_iter_text = payload

            # Flush any remaining holdback from parser
            for etype, edata in parser.flush():
                if etype == "text":
                    await send_msg(client_writer, {"type": "chunk", "delta": edata})
                    continuation_text += edata

            # Reset parser for next iteration
            parser = StreamParser()

            if not interrupts_found:
                log.info("no interrupts in stream iteration %d — loop complete.", iteration)
                full_text_all = continuation_text
                break

            log.info("stream iteration %d found %d interrupt(s)", iteration, len(interrupts_found))

            # Build continuation messages for next iteration
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


# ---------------------------------------------------------------------------
# Connection handler
# ---------------------------------------------------------------------------


async def handle_connection(
    service: AgrService,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
):
    try:
        msg = await recv_msg(reader)
        msg_type = msg.get("type")

        log.info("request type=%s", msg_type)

        if msg_type == "ping":
            await send_msg(writer, {"type": "pong"})

        elif msg_type == "run":
            messages = msg.get("messages", [])
            temperature = float(msg.get("temperature", 0.7))
            top_p = float(msg.get("top_p", 1.0))
            max_iterations = int(
                msg.get("max_iterations", service.default_max_iter)
            )

            log.info("run: %d messages, temp=%.2f, top_p=%.2f, max_iter=%d",
                     len(messages), temperature, top_p, max_iterations)

            try:
                await service.run_streaming(
                    messages, temperature, top_p, max_iterations, writer,
                )
            except Exception as e:
                log.error("agentic runtime error: %s", e)
                await send_msg(writer, {
                    "type": "error",
                    "message": f"Agentic runtime error: {e}",
                })

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
    log = setup_logging("zino.agr", config)

    service = AgrService(config)

    addr = config.get("agr", {}).get("socket", "/run/zino/agr.sock")
    server = await start_server(
        lambda r, w: handle_connection(service, r, w), addr, log,
    )
    await server.serve_forever()


log = None  # set in main()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="zino-agr: agentic runtime service")
    parser.add_argument("--config", "-c",
                        default=os.environ.get("ZINO_CONFIG", "ZINO.toml"))
    args = parser.parse_args()
    asyncio.run(main(args.config))
