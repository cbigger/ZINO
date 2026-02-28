#!/usr/bin/env python3
"""
zino-cli — minimal test client for zino-daemon.

Usage:
  python3 zino-cli.py "your message here"
  python3 zino-cli.py --stream "your message here"
  python3 zino-cli.py --config /path/to/ZINO.toml "your message here"
"""

import asyncio
import os
import sys
import tomllib
from pathlib import Path

from zino_common import send_msg, recv_msg, open_uds


def load_socket(config_path: str) -> str:
    p = Path(config_path)
    if not p.exists():
        return "/run/zino/daemon.sock"
    with open(p, "rb") as f:
        config = tomllib.load(f)
    return config.get("daemon", {}).get("socket", "/run/zino/daemon.sock")


async def send_request(socket_path: str, message: str, stream: bool, channel_id: str | None):
    reader, writer = await open_uds(socket_path)

    payload = {"message": message, "stream": stream}
    if channel_id:
        payload["channel_id"] = channel_id

    await send_msg(writer, payload)

    if stream:
        while True:
            packet = await recv_msg(reader)
            ptype = packet.get("type")
            if ptype == "chunk":
                print(packet.get("delta", ""), end="", flush=True)
            elif ptype == "done":
                print()  # final newline
                break
            elif ptype == "error":
                print(f"\nError: {packet.get('message')}", file=sys.stderr)
                break
            else:
                print(f"\nUnknown packet: {packet}", file=sys.stderr)
                break
    else:
        packet = await recv_msg(reader)
        ptype = packet.get("type")
        if ptype == "response":
            print(packet.get("content", ""))
        elif ptype == "error":
            print(f"Error: {packet.get('message')}", file=sys.stderr)
        else:
            print(f"Unexpected: {packet}", file=sys.stderr)

    writer.close()
    await writer.wait_closed()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="zino-cli: test client for zino-daemon")
    parser.add_argument("message", nargs="*", help="Message to send")
    parser.add_argument("--stream",     "-s", action="store_true")
    parser.add_argument("--channel",    "-ch", default=None)
    parser.add_argument("--config",     "-c",  default=os.environ.get("ZINO_CONFIG", "ZINO.toml"))
    args = parser.parse_args()

    message = " ".join(args.message) if args.message else "What is today's date?"
    socket_path = load_socket(args.config)

    asyncio.run(send_request(socket_path, message, args.stream, args.channel))
