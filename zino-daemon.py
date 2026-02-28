#!/usr/bin/env python3
"""
zino-daemon — main entrypoint and coordinator.

Responsibilities:
  - Receives (message, channel_id) from clients over UDS.
  - Assembles the most minimal valid messages array:
      - System prompt from zino-sys if available; omitted if not.
      - User message.
  - Queries zino-rtr capabilities on startup; routes accordingly.
  - Returns the model response to the client (streaming or collected).
  - Acts as the upgrade path for ctx/mem services when they exist.

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

from zino_common import send_msg, recv_msg, open_uds

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
# Service probes
# ---------------------------------------------------------------------------


async def probe_rtr(rtr_socket: str) -> dict:
    """Ask zino-rtr for its capabilities. Returns {} on failure."""
    try:
        reader, writer = await open_uds(rtr_socket)
        await send_msg(writer, {"type": "capabilities"})
        caps = await recv_msg(reader)
        writer.close()
        await writer.wait_closed()
        return caps
    except Exception as e:
        print(f"[daemon] Could not probe rtr at {rtr_socket}: {e}", file=sys.stderr)
        return {}


async def fetch_system_prompt(sys_socket: str) -> str | None:
    """
    Request the current built system prompt from zino-sys.
    Returns None if zino-sys is unavailable — daemon continues without a system prompt.
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
        print(f"[daemon] Could not reach zino-sys at {sys_socket}: {e} — bare mode.", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def assemble_messages(user_message: str, system_prompt: str | None) -> list[dict]:
    """
    Assemble the minimal valid messages array.

      [system]      ← from zino-sys if available; omitted if not
      [ctx/memory]  ← from zino-ctx/zino-mem when available; omitted for now
      [user]        ← always present
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # TODO: inject hard memory / context history here (zino-ctx, zino-mem)

    messages.append({"role": "user", "content": user_message})
    return messages


# ---------------------------------------------------------------------------
# RTR dispatch
# ---------------------------------------------------------------------------


async def dispatch_to_rtr(
    rtr_socket: str,
    messages: list[dict],
    temperature: float,
    top_p: float,
    want_stream: bool,
    rtr_caps: dict,
    client_writer: asyncio.StreamWriter,
):
    """
    Send assembled messages to zino-rtr.
    Forward the response to the client.
    Streaming is only requested if the client wants it AND rtr is capable.
    """
    do_stream = want_stream and rtr_caps.get("streaming", False)

    try:
        reader, writer = await open_uds(rtr_socket)
    except Exception as e:
        await send_msg(client_writer, {"type": "error", "message": f"Could not connect to rtr: {e}"})
        return

    try:
        await send_msg(writer, {
            "type":        "infer",
            "messages":    messages,
            "temperature": temperature,
            "top_p":       top_p,
            "stream":      do_stream,
        })

        if do_stream:
            while True:
                packet = await recv_msg(reader)
                await send_msg(client_writer, packet)
                if packet.get("type") in ("done", "error"):
                    break
        else:
            packet = await recv_msg(reader)
            await send_msg(client_writer, packet)

    except Exception as e:
        await send_msg(client_writer, {"type": "error", "message": f"rtr dispatch error: {e}"})
    finally:
        writer.close()
        await writer.wait_closed()


# ---------------------------------------------------------------------------
# Connection handler
# ---------------------------------------------------------------------------


async def handle_client(
    config: dict,
    sys_socket: str,
    rtr_socket: str,
    rtr_caps: dict,
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

        want_stream = bool(msg.get("stream", False))

        # TODO: retrieve channel history from zino-mem using msg.get("channel_id")

        agent       = config.get("agent", {})
        temperature = float(agent.get("temperature", 0.7))
        top_p       = float(agent.get("top_p", 1.0))

        # Fetch system prompt from zino-sys; None = bare mode (not an error)
        system_prompt = await fetch_system_prompt(sys_socket)

        messages = assemble_messages(user_message, system_prompt)

        await dispatch_to_rtr(
            rtr_socket, messages, temperature, top_p,
            want_stream, rtr_caps, writer,
        )

    except asyncio.IncompleteReadError:
        pass
    except Exception as e:
        print(f"[daemon] Handler error ({peer}): {e}", file=sys.stderr)
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

    daemon_cfg  = config.get("daemon", {})
    socket_path = daemon_cfg.get("socket", "/run/zino/daemon.sock")
    rtr_socket  = config.get("rtr", {}).get("socket", "/run/zino/rtr.sock")
    sys_socket  = config.get("sys", {}).get("socket", "/run/zino/sys.sock")

    # Probe rtr capabilities
    rtr_caps = await probe_rtr(rtr_socket)
    if rtr_caps:
        print(f"[daemon] rtr capabilities: {rtr_caps}")
    else:
        print("[daemon] Warning: could not reach rtr at startup. Will retry per request.")

    # Check sys availability (non-fatal)
    test_prompt = await fetch_system_prompt(sys_socket)
    if test_prompt:
        print("[daemon] zino-sys available.")
    else:
        print("[daemon] zino-sys unavailable — running in bare mode.")

    Path(socket_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        Path(socket_path).unlink()
    except FileNotFoundError:
        pass

    server = await asyncio.start_unix_server(
        lambda r, w: handle_client(config, sys_socket, rtr_socket, rtr_caps, r, w),
        path=socket_path,
    )

    print(f"[daemon] Listening on {socket_path}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="zino-daemon: main coordinator")
    parser.add_argument("--config", "-c", default=os.environ.get("ZINO_CONFIG", "ZINO.toml"))
    args = parser.parse_args()
    asyncio.run(main(args.config))
