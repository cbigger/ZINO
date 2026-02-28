#!/usr/bin/env python3
"""
zino-cli — minimal test client for zino-daemon.

Usage:
  python3 zino-cli.py "your message here"
  python3 zino-cli.py --config /path/to/ZINO.toml "your message here"
"""

import asyncio
import os
import sys
import tomllib
from pathlib import Path

from zino_common import send_msg, recv_msg, open_uds

SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


async def spinner(description: str):
    i = 0
    try:
        while True:
            print(f"\r  {SPINNER_CHARS[i % len(SPINNER_CHARS)]} {description}", end="", flush=True)
            await asyncio.sleep(0.1)
            i += 1
    except asyncio.CancelledError:
        print(f"\r  ✓ {description}", flush=True)


def load_socket(config_path: str) -> str:
    p = Path(config_path)
    if not p.exists():
        return "/run/zino/daemon.sock"
    with open(p, "rb") as f:
        config = tomllib.load(f)
    return config.get("daemon", {}).get("socket", "/run/zino/daemon.sock")


async def send_request(socket_path: str, message: str, channel_id: str | None):
    reader, writer = await open_uds(socket_path)

    payload = {"message": message}
    if channel_id:
        payload["channel_id"] = channel_id

    await send_msg(writer, payload)

    spinner_task = None
    try:
        while True:
            packet = await recv_msg(reader)
            ptype = packet.get("type")
            if ptype == "chunk":
                print(packet.get("delta", ""), end="", flush=True)
            elif ptype == "tool_start":
                desc = packet.get("description", "working...")
                spinner_task = asyncio.create_task(spinner(desc))
            elif ptype == "tool_done":
                if spinner_task:
                    spinner_task.cancel()
                    try:
                        await spinner_task
                    except asyncio.CancelledError:
                        pass
                    spinner_task = None
            elif ptype == "done":
                print()  # final newline
                break
            elif ptype == "error":
                print(f"\nError: {packet.get('message')}", file=sys.stderr)
                break
            else:
                print(f"\nUnknown packet: {packet}", file=sys.stderr)
                break
    finally:
        if spinner_task:
            spinner_task.cancel()
            try:
                await spinner_task
            except asyncio.CancelledError:
                pass
        writer.close()
        await writer.wait_closed()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="zino-cli: test client for zino-daemon")
    parser.add_argument("message", nargs="*", help="Message to send")
    parser.add_argument("--channel",    "-ch", default=None)
    parser.add_argument("--config",     "-c",  default=os.environ.get("ZINO_CONFIG", "ZINO.toml"))
    args = parser.parse_args()

    message = " ".join(args.message) if args.message else "What is today's date?"
    socket_path = load_socket(args.config)

    asyncio.run(send_request(socket_path, message, args.channel))
