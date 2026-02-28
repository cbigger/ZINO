"""
zino_common.py — shared wire protocol helpers for inter-service communication.

Supports two transports:
  - UDS (Unix Domain Socket): 4-byte big-endian uint32 length prefix + UTF-8 JSON.
  - HTTP: aiohttp-based, using Server-Sent Events (SSE) for responses.

Transport is selected based on address format:
  - "/run/zino/rtr.sock"      → UDS (default)
  - "http://0.0.0.0:8001"     → HTTP

All services speak through send_msg() / recv_msg() which dispatch on
the type of reader/writer, so handler code is transport-agnostic.
"""

import asyncio
import json
import logging
import os
import struct

# ---------------------------------------------------------------------------
# Lazy aiohttp import (only needed when HTTP addresses are configured)
# ---------------------------------------------------------------------------

_aiohttp = None


def _ensure_aiohttp():
    global _aiohttp
    if _aiohttp is None:
        try:
            import aiohttp as _mod
            _aiohttp = _mod
        except ImportError:
            raise ImportError(
                "aiohttp is required for HTTP transport. "
                "Install it with: pip install aiohttp>=3.9.0"
            )
    return _aiohttp


# ---------------------------------------------------------------------------
# Address helpers
# ---------------------------------------------------------------------------


def is_http_address(addr: str) -> bool:
    """Return True if addr is an HTTP URL."""
    return addr.startswith("http://")


def parse_address(addr: str):
    """
    Parse an address string into a transport descriptor.

    Returns:
      ("unix", path)           for UDS addresses
      ("http", host, port)     for HTTP addresses
    """
    if is_http_address(addr):
        from urllib.parse import urlparse
        u = urlparse(addr)
        host = u.hostname or "0.0.0.0"
        port = u.port or 80
        return ("http", host, port)
    return ("unix", addr)


# ---------------------------------------------------------------------------
# Length-prefixed JSON framing (UDS wire format)
# ---------------------------------------------------------------------------

HEADER = struct.Struct("!I")   # 4-byte big-endian unsigned int


# ---------------------------------------------------------------------------
# HTTP adapter classes — duck-type asyncio.StreamReader / StreamWriter
# ---------------------------------------------------------------------------


class HttpServerReader:
    """Wraps a parsed POST JSON body; recv_msg() returns it once."""

    def __init__(self, body: dict):
        self._body = body
        self._consumed = False

    async def recv(self) -> dict:
        if self._consumed:
            raise asyncio.IncompleteReadError(b"", None)
        self._consumed = True
        return self._body


class HttpServerWriter:
    """Wraps an aiohttp.StreamResponse; send_msg() writes SSE events."""

    def __init__(self, response, request):
        self._response = response
        self._request = request

    async def send(self, payload: dict):
        line = "data: " + json.dumps(payload) + "\n\n"
        await self._response.write(line.encode("utf-8"))

    def get_extra_info(self, key, default=None):
        if key == "peername":
            return self._request.remote
        return default

    def close(self):
        pass

    async def wait_closed(self):
        pass


class HttpClientReader:
    """Reads SSE events from an aiohttp.ClientResponse."""

    def __init__(self):
        self._response = None  # set by paired writer after POST

    async def recv(self) -> dict:
        if self._response is None:
            raise RuntimeError("HttpClientReader: no response set (send first)")
        async for raw_line in self._response.content:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
            if line.startswith("data: "):
                return json.loads(line[6:])
            # skip empty lines and other SSE fields
        raise asyncio.IncompleteReadError(b"", None)


class HttpClientWriter:
    """
    On first send_msg(), POSTs JSON to the URL and stores the response
    for the paired HttpClientReader.  Subsequent sends are no-ops
    (single-request protocol).
    """

    def __init__(self, url: str, client_reader: 'HttpClientReader'):
        self._url = url
        self._client_reader = client_reader
        self._session = None
        self._sent = False

    async def send(self, payload: dict):
        if self._sent:
            return
        self._sent = True
        aiohttp = _ensure_aiohttp()
        self._session = aiohttp.ClientSession()
        resp = await self._session.post(
            self._url,
            json=payload,
            headers={"Accept": "text/event-stream"},
        )
        self._client_reader._response = resp

    def close(self):
        pass

    async def wait_closed(self):
        if self._session:
            await self._session.close()
            self._session = None


# ---------------------------------------------------------------------------
# send_msg / recv_msg — 3-way dispatch
# ---------------------------------------------------------------------------


async def send_msg(writer, payload: dict) -> None:
    """
    Send a JSON payload via the appropriate transport.

    Dispatches based on writer type:
      - HttpServerWriter  → SSE event to response stream
      - HttpClientWriter  → POST JSON to URL (first call only)
      - StreamWriter      → length-prefixed binary protocol (UDS)
    """
    if isinstance(writer, (HttpServerWriter, HttpClientWriter)):
        await writer.send(payload)
    else:
        data = json.dumps(payload).encode("utf-8")
        writer.write(HEADER.pack(len(data)) + data)
        await writer.drain()


async def recv_msg(reader) -> dict:
    """
    Read one JSON message via the appropriate transport.

    Dispatches based on reader type:
      - HttpServerReader  → return stored POST body dict
      - HttpClientReader  → parse next SSE data: line
      - StreamReader      → length-prefixed binary read (UDS)
    """
    if isinstance(reader, (HttpServerReader, HttpClientReader)):
        return await reader.recv()
    else:
        header = await reader.readexactly(HEADER.size)
        (length,) = HEADER.unpack(header)
        data = await reader.readexactly(length)
        return json.loads(data.decode("utf-8"))


# ---------------------------------------------------------------------------
# open_connection — unified client connection
# ---------------------------------------------------------------------------


async def open_connection(addr: str):
    """
    Open a connection to a ZINO service.

    Returns (reader, writer) that work with send_msg() / recv_msg().

    addr formats:
      "/run/zino/rtr.sock"      → UDS connection
      "http://127.0.0.1:8001"   → HTTP connection (paired adapters)
    """
    parsed = parse_address(addr)
    if parsed[0] == "http":
        reader = HttpClientReader()
        writer = HttpClientWriter(addr, reader)
        return reader, writer
    else:
        return await asyncio.open_unix_connection(parsed[1])


async def open_uds(path: str):
    """Deprecated: use open_connection() instead."""
    return await open_connection(path)


# ---------------------------------------------------------------------------
# start_server — unified server startup
# ---------------------------------------------------------------------------


class UDSServerHandle:
    """Wraps asyncio.Server for unified lifecycle."""

    def __init__(self, server, socket_path):
        self._server = server
        self._socket_path = socket_path

    async def serve_forever(self):
        async with self._server:
            await self._server.serve_forever()

    async def cleanup(self):
        self._server.close()
        await self._server.wait_closed()


class HTTPServerHandle:
    """Wraps aiohttp.web.AppRunner for unified lifecycle."""

    def __init__(self, runner, site):
        self._runner = runner
        self._site = site

    async def serve_forever(self):
        # Block indefinitely — aiohttp serves in the background
        await asyncio.Event().wait()

    async def cleanup(self):
        await self._runner.cleanup()


async def start_server(handler, addr: str, log: logging.Logger):
    """
    Start a ZINO service server.

    handler: async callable(reader, writer) — the service's connection handler.
    addr:    socket path or http:// URL.
    log:     logger instance.

    Returns a ServerHandle with serve_forever() and cleanup().

    For UDS: creates parent dir, removes stale socket, starts unix server.
    For HTTP: creates aiohttp app with POST / route, starts TCP site.
    """
    from pathlib import Path

    parsed = parse_address(addr)

    if parsed[0] == "http":
        _, host, port = parsed
        aiohttp = _ensure_aiohttp()
        from aiohttp import web

        async def http_handler(request: web.Request) -> web.StreamResponse:
            try:
                body = await request.json()
            except Exception:
                return web.Response(status=400, text="Invalid JSON body")

            response = web.StreamResponse(
                status=200,
                reason="OK",
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
            await response.prepare(request)

            reader = HttpServerReader(body)
            writer = HttpServerWriter(response, request)

            try:
                await handler(reader, writer)
            except Exception as e:
                log.error("HTTP handler error: %s", e)
                try:
                    err_line = "data: " + json.dumps({"type": "error", "message": str(e)}) + "\n\n"
                    await response.write(err_line.encode("utf-8"))
                except Exception:
                    pass

            try:
                await response.write_eof()
            except Exception:
                pass
            return response

        app = web.Application()
        app.router.add_post("/", http_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        log.info("listening on http://%s:%d", host, port)
        return HTTPServerHandle(runner, site)

    else:
        socket_path = parsed[1]
        Path(socket_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            Path(socket_path).unlink()
        except FileNotFoundError:
            pass

        server = await asyncio.start_unix_server(handler, path=socket_path)
        log.info("listening on %s", socket_path)
        return UDSServerHandle(server, socket_path)


# ---------------------------------------------------------------------------
# Logging setup — shared by all services
# ---------------------------------------------------------------------------


def setup_logging(name: str, config: dict | None = None) -> logging.Logger:
    """
    Configure and return a logger for a ZINO service.

    Level is determined by (highest priority first):
      1. ZINO_LOG_LEVEL environment variable
      2. [logging] level in ZINO.toml
      3. "INFO" default

    Output goes to stderr (which systemd captures into the journal).
    """
    level_str = (
        os.environ.get("ZINO_LOG_LEVEL")
        or (config or {}).get("logging", {}).get("level")
        or "INFO"
    ).upper()

    level = getattr(logging, level_str, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(name)s %(levelname)s %(message)s",
        force=True,
    )

    log = logging.getLogger(name)
    log.setLevel(level)
    log.info("log level: %s", level_str)
    return log
