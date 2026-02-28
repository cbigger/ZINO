"""
zino_common.py — shared wire protocol helpers for inter-service UDS communication.

Wire format: 4-byte big-endian uint32 length prefix + UTF-8 JSON payload.
All services speak this protocol.
"""

import asyncio
import json
import struct

# ---------------------------------------------------------------------------
# Length-prefixed JSON framing
# ---------------------------------------------------------------------------

HEADER = struct.Struct("!I")   # 4-byte big-endian unsigned int


async def send_msg(writer: asyncio.StreamWriter, payload: dict) -> None:
    """Encode payload as JSON and send with a 4-byte length prefix."""
    data = json.dumps(payload).encode("utf-8")
    writer.write(HEADER.pack(len(data)) + data)
    await writer.drain()


async def recv_msg(reader: asyncio.StreamReader) -> dict:
    """Read one length-prefixed JSON message from reader."""
    header = await reader.readexactly(HEADER.size)
    (length,) = HEADER.unpack(header)
    data = await reader.readexactly(length)
    return json.loads(data.decode("utf-8"))


# ---------------------------------------------------------------------------
# Convenience: open a UDS connection and return (reader, writer)
# ---------------------------------------------------------------------------

async def open_uds(path: str):
    return await asyncio.open_unix_connection(path)
