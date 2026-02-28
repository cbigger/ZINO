#!/usr/bin/env python3
"""
zino-rtr — LLM router daemon.

Responsibilities:
  - Maintains the LLM API connection (URL, model, API key).
  - Accepts assembled prompt payloads from zino-daemon over UDS.
  - Streams or collects the model response based on configuration.
  - Advertises capabilities to callers on request.

UDS socket: /run/zino/rtr.sock (configurable)

Inbound message types:
  {"type": "capabilities"}
      → {"streaming": bool, "model": str}

  {"type": "infer", "messages": [...], "temperature": float, "top_p": float, "stream": bool}
      If stream=True and streaming is capable:
          → N × {"type": "chunk",    "delta": str}
          →   1 × {"type": "done",     "full_text": str}
      Otherwise:
          →   1 × {"type": "response", "content": str}

  {"type": "ping"}
      → {"type": "pong"}
"""

import asyncio
import json
import os
import sys
import tomllib
from pathlib import Path

from dotenv import load_dotenv
import openai

from zino_common import send_msg, recv_msg, HEADER

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PLACEHOLDER = ""


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[rtr] Config not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p, "rb") as f:
        return tomllib.load(f)


def resolve_api_key(config: dict) -> str:
    key = config.get("key", "")
    if key:  # and key != _PLACEHOLDER:
        return key
    load_dotenv()
    key = os.getenv("LLM_API_KEY", "")
    if key:
        return key
    print("[rtr] No valid API key. Set LLM_API_KEY or provide it in ZINO.toml.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Router state
# ---------------------------------------------------------------------------

class Router:
    def __init__(self, config: dict):
        api     = config["api"]
        rtr_cfg = config.get("rtr", {})

        self.model     = config["service"]["model"]
        self.streaming = rtr_cfg.get("streaming", True)

        self.client = openai.OpenAI(
            api_key  = resolve_api_key(api),
            base_url = api["base_url"],
        )

        print(f"[rtr] model={self.model} streaming={self.streaming}")

    def capabilities(self) -> dict:
        return {"streaming": self.streaming, "model": self.model}

    def infer_collect(self, messages: list, temperature: float, top_p: float) -> str:
        """Non-streaming: collect full response and return."""
        resp = self.client.chat.completions.create(
            model       = self.model,
            messages    = messages,
            temperature = temperature,
            top_p       = top_p,
            stream      = False,
        )
        return resp.choices[0].message.content or ""

    def infer_stream(self, messages: list, temperature: float, top_p: float):
        """Streaming: yield text deltas, then the full assembled text."""
        full_text = ""
        stream = self.client.chat.completions.create(
            model       = self.model,
            messages    = messages,
            temperature = temperature,
            top_p       = top_p,
            stream      = True,
        )
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta is None:
                    continue
                full_text += delta
                yield "chunk", delta
        finally:
            try:
                stream.close()
            except Exception:
                pass
        yield "done", full_text


# ---------------------------------------------------------------------------
# Connection handler
# ---------------------------------------------------------------------------

async def handle_connection(router: Router, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    peer = writer.get_extra_info("peername", "<unknown>")
    try:
        msg = await recv_msg(reader)
        msg_type = msg.get("type")

        print(json.dumps(msg, indent=4))

        if msg_type == "ping":
            await send_msg(writer, {"type": "pong"})

        elif msg_type == "capabilities":
            await send_msg(writer, router.capabilities())

        elif msg_type == "infer":
            messages    = msg.get("messages", [])
            temperature = float(msg.get("temperature", 0.7))
            top_p       = float(msg.get("top_p", 1.0))
            want_stream = bool(msg.get("stream", False))

            if want_stream and router.streaming:
                # Run blocking stream generator in a thread
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue = asyncio.Queue()

                def _produce():
                    try:
                        for kind, text in router.infer_stream(messages, temperature, top_p):
                            queue.put_nowait((kind, text))
                    except Exception as e:
                        queue.put_nowait(("error", str(e)))

                await loop.run_in_executor(None, _produce)

                while True:
                    kind, text = await queue.get()
                    if kind == "chunk":
                        await send_msg(writer, {"type": "chunk", "delta": text})
                    elif kind == "done":
                        await send_msg(writer, {"type": "done", "full_text": text})
                        break
                    elif kind == "error":
                        await send_msg(writer, {"type": "error", "message": text})
                        break
            else:
                loop = asyncio.get_running_loop()
                try:
                    content = await loop.run_in_executor(
                        None, router.infer_collect, messages, temperature, top_p
                    )
                    await send_msg(writer, {"type": "response", "content": content})
                except Exception as e:
                    await send_msg(writer, {"type": "error", "message": str(e)})
        else:
            await send_msg(writer, {"type": "error", "message": f"Unknown message type: {msg_type}"})

    except asyncio.IncompleteReadError:
        pass
    except Exception as e:
        print(f"[rtr] Handler error ({peer}): {e}", file=sys.stderr)
    finally:
        writer.close()
        await writer.wait_closed()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(config_path: str):
    config = load_config(config_path)
    router = Router(config)

    socket_path = config.get("rtr", {}).get("socket", "/run/zino/rtr.sock")
    Path(socket_path).parent.mkdir(parents=True, exist_ok=True)

    # Remove stale socket
    try:
        Path(socket_path).unlink()
    except FileNotFoundError:
        pass

    server = await asyncio.start_unix_server(
        lambda r, w: handle_connection(router, r, w),
        path=socket_path,
    )

    print(f"[rtr] Listening on {socket_path}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="zino-rtr: LLM router daemon")
    parser.add_argument("--config", "-c", default=os.environ.get("ZINO_CONFIG", "ZINO.toml"))
    args = parser.parse_args()
    asyncio.run(main(args.config))
