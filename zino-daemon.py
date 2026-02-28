#!/usr/bin/env python3
"""
zino-daemon — main entrypoint and coordinator.

Responsibilities:
  - Receives (message, channel_id) from clients over UDS.
  - Assembles the full messages array:
      - System prompt from zino-sys (optional).
      - Context history from zino-ctx (optional): examples, hard memories, chat history.
      - User message.
  - Routes through zino-agr for agentic loop (tool/skill execution), or
    falls back to direct zino-rtr dispatch when agr is unavailable.
  - Stores chat history via zino-mem (optional).
  - Returns the model response to the client (streaming or collected).

All service connections are attempted live per-request.  If a service is
unreachable the daemon degrades gracefully — no startup probe cache that
can go stale.

UDS socket: /run/zino/daemon.sock (configurable)

Inbound message from client:
  {"message": str, "channel_id": str (optional), "stream": bool (optional)}

Outbound to client:
  Streaming:    N × {"type": "chunk",    "delta":     str}
                  1 × {"type": "done",     "full_text": str}
  Collected:      1 × {"type": "response", "content":   str}
  Error:          1 × {"type": "error",    "message":   str}
"""

import asyncio
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
        print(f"[daemon] Config not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Service interactions (all tolerant of unavailable services)
# ---------------------------------------------------------------------------


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


async def fetch_system_prompt(sys_socket: str) -> str | None:
    """
    Request the current built system prompt from zino-sys.
    Returns None if zino-sys is unavailable — daemon continues in bare mode.
    """
    try:
        reader, writer = await open_uds(sys_socket)
        await send_msg(writer, {"type": "get"})
        msg = await recv_msg(reader)
        writer.close()
        await writer.wait_closed()
        if msg.get("type") == "system_prompt":
            content = msg.get("content", "")
            return content if content else None
        return None
    except Exception as e:
        log.warning("could not reach zino-sys: %s — bare mode.", e)
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
        log.warning("could not reach zino-ctx: %s — no context.", e)
        return []


async def store_history(
    mem_socket: str, channel_id: str, user_msg: str, assistant_msg: str,
):
    """Append a user+assistant exchange to channel history in zino-mem."""
    try:
        reader, writer = await open_uds(mem_socket)
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
        log.warning("could not store history in zino-mem: %s", e)


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

      [system]      ← from zino-sys if available
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
# AGR dispatch
# ---------------------------------------------------------------------------


async def dispatch_to_agr(
    agr_socket: str,
    messages: list[dict],
    temperature: float,
    top_p: float,
    max_iterations: int,
    want_stream: bool,
    client_writer: asyncio.StreamWriter,
) -> str | None:
    """
    Send assembled messages to zino-agr for agentic processing.

    When want_stream=True, agr streams chunk/done/error packets which are
    forwarded directly to client_writer in real-time.

    When want_stream=False, chunks are collected internally and sent as a
    single {"type": "response"} packet.

    Returns the full response text (for history storage), or None on failure.
    """
    try:
        reader, writer = await open_uds(agr_socket)
    except Exception as e:
        log.info("agr not available: %s — falling back to rtr.", e)
        return None

    try:
        await send_msg(writer, {
            "type":           "run",
            "messages":       messages,
            "temperature":    temperature,
            "top_p":          top_p,
            "max_iterations": max_iterations,
            "stream":         want_stream,
        })

        if want_stream:
            # Forward streaming packets from agr directly to client
            full_text = None
            while True:
                packet = await recv_msg(reader)
                ptype = packet.get("type")
                if ptype == "chunk":
                    await send_msg(client_writer, packet)
                elif ptype == "done":
                    await send_msg(client_writer, packet)
                    full_text = packet.get("full_text", "")
                    log.info("agr streaming done: %d chars", len(full_text))
                    break
                elif ptype == "error":
                    await send_msg(client_writer, packet)
                    log.error("agr streaming error: %s", packet.get("message"))
                    break
                else:
                    log.warning("unexpected agr packet type: %s", ptype)
                    break
            return full_text
        else:
            # Non-streaming: read the single result packet
            packet = await recv_msg(reader)
            if packet.get("type") == "result":
                content = packet.get("content", "")
                await send_msg(client_writer, {"type": "response", "content": content})
                return content
            if packet.get("type") == "error":
                log.error("agr error: %s", packet.get("message"))
                await send_msg(client_writer, packet)
            return None

    except Exception as e:
        log.error("agr dispatch error: %s", e)
        try:
            await send_msg(client_writer, {
                "type": "error",
                "message": f"agr dispatch error: {e}",
            })
        except Exception:
            pass
        return None
    finally:
        writer.close()
        await writer.wait_closed()


# ---------------------------------------------------------------------------
# RTR dispatch (fallback when agr is unavailable)
# ---------------------------------------------------------------------------


async def dispatch_to_rtr(
    rtr_socket: str,
    messages: list[dict],
    temperature: float,
    top_p: float,
    want_stream: bool,
    rtr_caps: dict,
    client_writer: asyncio.StreamWriter,
) -> str | None:
    """
    Send assembled messages to zino-rtr.
    Forward the response to the client.
    Returns the full response text (for history storage), or None on error.
    """
    do_stream = want_stream and rtr_caps.get("streaming", False)

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
            await send_msg(client_writer, packet)
            if packet.get("type") == "response":
                return packet.get("content", "")
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

        channel_id  = msg.get("channel_id")
        want_stream = bool(msg.get("stream", False))

        log.info("request: msg_len=%d channel=%s stream=%s",
                 len(user_message), channel_id, want_stream)
        log.debug("user message: %s", user_message)

        agent       = config.get("agent", {})
        temperature = float(agent.get("temperature", 0.7))
        top_p       = float(agent.get("top_p", 1.0))
        max_iter    = int(config.get("agr", {}).get("max_iterations", 5))

        # Lazy re-probe rtr capabilities if the cache is empty
        if not rtr_caps_cache:
            fresh = await probe_rtr(sockets["rtr"])
            if fresh:
                rtr_caps_cache.update(fresh)
                log.info("rtr capabilities acquired: %s", rtr_caps_cache)

        # 1. System prompt from zino-sys (optional — tries every request)
        system_prompt = await fetch_system_prompt(sockets["sys"])
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

        # 4. Route through agr (agentic loop), fall back to direct rtr
        response_content = await dispatch_to_agr(
            sockets["agr"], messages, temperature, top_p, max_iter,
            want_stream, writer,
        )
        if response_content is None:
            log.info("using direct rtr dispatch (stream=%s)", want_stream)
            response_content = await dispatch_to_rtr(
                sockets["rtr"], messages, temperature, top_p,
                want_stream, rtr_caps_cache, writer,
            )

        # 5. Store history in zino-mem (optional — tries every request)
        if channel_id and response_content:
            await store_history(
                sockets["mem"], channel_id, user_message, response_content,
            )
            log.info("history stored for channel=%s", channel_id)

        log.info("request complete: response=%s",
                 f"{len(response_content)} chars" if response_content else "none")

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
        "sys":    config.get("sys", {}).get("socket", "/run/zino/sys.sock"),
        "exc":    config.get("exc", {}).get("socket", "/run/zino/exc.sock"),
        "mem":    config.get("mem", {}).get("socket", "/run/zino/mem.sock"),
        "ctx":    config.get("ctx", {}).get("socket", "/run/zino/ctx.sock"),
        "agr":    config.get("agr", {}).get("socket", "/run/zino/agr.sock"),
    }

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
        lambda r, w: handle_client(config, sockets, rtr_caps_cache, r, w),
        path=socket_path,
    )

    log.info("listening on %s", socket_path)
    async with server:
        await server.serve_forever()


log = None  # set in main()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="zino-daemon: main coordinator")
    parser.add_argument("--config", "-c",
                        default=os.environ.get("ZINO_CONFIG", "ZINO.toml"))
    args = parser.parse_args()
    asyncio.run(main(args.config))
