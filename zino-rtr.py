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

from zino_common import send_msg, recv_msg, HEADER, start_server, setup_logging

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

        log.info("model=%s streaming=%s base_url=%s",
                 self.model, self.streaming, api["base_url"])

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

        log.info("request type=%s", msg_type)

        if msg_type == "ping":
            await send_msg(writer, {"type": "pong"})

        elif msg_type == "capabilities":
            await send_msg(writer, router.capabilities())

        elif msg_type == "infer":
            messages    = msg.get("messages", [])
            temperature = float(msg.get("temperature", 0.7))
            top_p       = float(msg.get("top_p", 1.0))
            want_stream = bool(msg.get("stream", False))

            log.info("infer: %d message(s), temperature=%.2f, top_p=%.2f, stream=%s",
                     len(messages), temperature, top_p, want_stream)
            log.debug("infer payload:\n%s", json.dumps(messages, indent=2))

            if want_stream and router.streaming:
                # Run blocking stream generator in a thread, consume concurrently
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue = asyncio.Queue()

                def _produce():
                    try:
                        for kind, text in router.infer_stream(messages, temperature, top_p):
                            loop.call_soon_threadsafe(queue.put_nowait, (kind, text))
                    except Exception as e:
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))

                # Start producer in thread pool — do NOT await yet
                producer = loop.run_in_executor(None, _produce)

                # Consumer runs concurrently, forwarding chunks as they arrive
                while True:
                    kind, text = await queue.get()
                    if kind == "chunk":
                        await send_msg(writer, {"type": "chunk", "delta": text})
                    elif kind == "done":
                        await send_msg(writer, {"type": "done", "full_text": text})
                        log.info("infer complete (streamed): %d chars", len(text))
                        log.debug("full response:\n%s", text)
                        break
                    elif kind == "error":
                        await send_msg(writer, {"type": "error", "message": text})
                        log.error("infer stream error: %s", text)
                        break

                # Ensure producer thread is done
                await producer
            else:
                loop = asyncio.get_running_loop()
                try:
                    content = await loop.run_in_executor(
                        None, router.infer_collect, messages, temperature, top_p
                    )
                    await send_msg(writer, {"type": "response", "content": content})
                    log.info("infer complete (collected): %d chars", len(content))
                    log.debug("full response:\n%s", content)
                except Exception as e:
                    await send_msg(writer, {"type": "error", "message": str(e)})
                    log.error("infer collect error: %s", e)
        else:
            await send_msg(writer, {"type": "error", "message": f"Unknown message type: {msg_type}"})
            log.warning("unknown message type: %s", msg_type)

    except asyncio.IncompleteReadError:
        pass
    except Exception as e:
        log.error("handler error (%s): %s", peer, e)
    finally:
        writer.close()
        await writer.wait_closed()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(config_path: str):
    config = load_config(config_path)

    global log
    log = setup_logging("zino.rtr", config)

    router = Router(config)

    addr = config.get("rtr", {}).get("socket", "/run/zino/rtr.sock")
    server = await start_server(
        lambda r, w: handle_connection(router, r, w), addr, log,
    )
    await server.serve_forever()


log = None  # set in main()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="zino-rtr: LLM router daemon")
    parser.add_argument("--config", "-c", default=os.environ.get("ZINO_CONFIG", "ZINO.toml"))
    args = parser.parse_args()
    asyncio.run(main(args.config))
