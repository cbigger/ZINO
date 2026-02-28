#!/usr/bin/env python3
"""
zino-exc — execution service.

Responsibilities:
  - Executes code produced by tools and skills.
  - Enforces execution timeouts.
  - Runs optional static analysis (shellcheck) before execution.
  - Manages temporary files for code execution.

UDS socket: /run/zino/exc.sock (configurable)

Inbound message types:
  {"type": "execute", "executor": str, "code": str}
      → {"type": "result", "exit_code": int, "stdout": str, "stderr": str}

  {"type": "validate", "executor": str, "code": str}
      → {"type": "validation", "ok": bool, "errors": str}

  {"type": "capabilities"}
      → {"type": "capabilities", "executors": {name: {...}}, "timeout": int}

  {"type": "ping"}
      → {"type": "pong"}
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path

from zino_common import send_msg, recv_msg, start_server, setup_logging

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"[exc] Config not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Executor registry
# ---------------------------------------------------------------------------


class Executor:
    """Represents a single execution environment (bash, python3, etc.)."""

    def __init__(self, name: str, meta: dict):
        self.name = name
        self.command = meta.get("executor", name)
        self.file_ext = meta.get("file_extension", "")
        self.static_analysis = meta.get("static_analysis")
        self.has_validator = False

        if self.static_analysis:
            self.has_validator = shutil.which(self.static_analysis) is not None
            if self.has_validator:
                log.info("executor '%s': validator '%s' available.",
                         name, self.static_analysis)
            else:
                log.warning("executor '%s': validator '%s' not found.",
                            name, self.static_analysis)

    def info(self) -> dict:
        return {
            "command": self.command,
            "file_extension": self.file_ext,
            "static_analysis": self.static_analysis,
            "has_validator": self.has_validator,
        }


def load_executors(config: dict) -> dict[str, Executor]:
    """Load executor definitions from tool JSON files."""
    tools_dir = Path(config.get("tools", {}).get("dir", "tools"))
    executors: dict[str, Executor] = {}

    if not tools_dir.exists():
        log.warning("tools directory not found: %s", tools_dir)
        return executors

    for tool_file in sorted(tools_dir.glob("*.json")):
        with open(tool_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("meta", {})
        name = meta.get("executor", tool_file.stem)
        if name not in executors:
            executors[name] = Executor(name, meta)

    return executors


# ---------------------------------------------------------------------------
# Execution logic
# ---------------------------------------------------------------------------


class ExecService:
    def __init__(self, config: dict):
        self.config = config
        self.executors = load_executors(config)
        self.timeout = config.get("exc", {}).get("timeout", 30)
        log.info("loaded %d executor(s): %s",
                 len(self.executors), list(self.executors.keys()))
        log.info("execution timeout: %ds", self.timeout)

    def capabilities(self) -> dict:
        return {
            "executors": {name: ex.info() for name, ex in self.executors.items()},
            "timeout": self.timeout,
        }

    async def validate(self, executor_name: str, code: str) -> dict:
        """Run static analysis on code. Returns {"ok": bool, "errors": str}."""
        ex = self.executors.get(executor_name)
        if not ex:
            return {"ok": False, "errors": f"Unknown executor: {executor_name}"}

        if not ex.has_validator:
            return {"ok": True, "errors": ""}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=ex.file_ext, delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                ex.static_analysis, tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
            ok = proc.returncode == 0
            errors = (stdout.decode("utf-8", errors="replace") +
                      stderr.decode("utf-8", errors="replace")).strip()
            return {"ok": ok, "errors": errors}
        except asyncio.TimeoutError:
            return {"ok": False, "errors": "Validation timed out."}
        except Exception as e:
            return {"ok": False, "errors": str(e)}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def execute(self, executor_name: str, code: str) -> dict:
        """Execute code and return {"exit_code": int, "stdout": str, "stderr": str}."""
        ex = self.executors.get(executor_name)
        if not ex:
            return {"exit_code": -1, "stdout": "",
                    "stderr": f"Unknown executor: {executor_name}"}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=ex.file_ext, delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            if ex.file_ext == ".sh":
                os.chmod(tmp_path, 0o755)

            proc = await asyncio.create_subprocess_exec(
                ex.command, tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
            return {
                "exit_code": proc.returncode,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
            }
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return {"exit_code": -1, "stdout": "",
                    "stderr": f"Execution timed out after {self.timeout}s."}
        except Exception as e:
            return {"exit_code": -1, "stdout": "", "stderr": str(e)}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Connection handler
# ---------------------------------------------------------------------------


async def handle_connection(
    service: ExecService,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
):
    try:
        msg = await recv_msg(reader)
        msg_type = msg.get("type")

        log.info("request type=%s", msg_type)

        if msg_type == "ping":
            await send_msg(writer, {"type": "pong"})

        elif msg_type == "capabilities":
            await send_msg(writer, {"type": "capabilities", **service.capabilities()})

        elif msg_type == "validate":
            executor = msg.get("executor", "")
            code = msg.get("code", "")
            log.info("validate: executor=%s code_len=%d", executor, len(code))
            log.debug("validate code:\n%s", code)
            result = await service.validate(executor, code)
            log.info("validate result: ok=%s", result["ok"])
            await send_msg(writer, {"type": "validation", **result})

        elif msg_type == "execute":
            executor = msg.get("executor", "")
            code = msg.get("code", "")
            log.info("execute: executor=%s code_len=%d", executor, len(code))
            log.debug("execute code:\n%s", code)

            # Run validation first if validator is available (warnings only)
            ex = service.executors.get(executor)
            if ex and ex.has_validator:
                vresult = await service.validate(executor, code)
                if not vresult["ok"]:
                    log.warning("validation warning for %s: %s",
                                executor, vresult["errors"])

            result = await service.execute(executor, code)
            log.info("execute result: executor=%s exit_code=%d stdout_len=%d stderr_len=%d",
                     executor, result["exit_code"],
                     len(result["stdout"]), len(result["stderr"]))
            log.debug("execute stdout:\n%s", result["stdout"])
            if result["stderr"]:
                log.debug("execute stderr:\n%s", result["stderr"])
            await send_msg(writer, {"type": "result", **result})

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
    log = setup_logging("zino.exc", config)

    service = ExecService(config)

    addr = config.get("exc", {}).get("socket", "/run/zino/exc.sock")
    server = await start_server(
        lambda r, w: handle_connection(service, r, w), addr, log,
    )
    await server.serve_forever()


log = None  # set in main()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="zino-exc: execution service")
    parser.add_argument("--config", "-c",
                        default=os.environ.get("ZINO_CONFIG", "ZINO.toml"))
    args = parser.parse_args()
    asyncio.run(main(args.config))
