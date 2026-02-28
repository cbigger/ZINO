# ZINO Service Reference

Functional documentation for each ZINO component service. All services communicate over Unix domain sockets using a length-prefixed JSON wire protocol (4-byte big-endian uint32 length + UTF-8 JSON payload), implemented in `zino_common.py`.

Every service accepts `{"type": "ping"}` and responds with `{"type": "pong"}`. This is omitted from the individual sections below.

---

## zino-daemon

**File:** `zino-daemon.py`
**Socket:** `/run/zino/daemon.sock`
**Role:** Main entrypoint and coordinator. Receives client requests, assembles the prompt (system prompt + context + user message), routes through agr or rtr, and stores history.

### Inbound (from client)

```
{"message": str, "channel_id": str?, "stream": bool?}
```

| Field | Required | Description |
|---|---|---|
| `message` | yes | The user's message text |
| `channel_id` | no | Channel identifier for history storage/retrieval |
| `stream` | no | If `true`, response streams as chunks. Default `false` |

### Outbound (to client)

**Streaming** (`stream: true`):
```
N x {"type": "chunk", "delta": str}
1 x {"type": "done", "full_text": str}
```

**Collected** (`stream: false`):
```
1 x {"type": "response", "content": str}
```

**Error** (either mode):
```
1 x {"type": "error", "message": str}
```

### Internal request flow

1. Fetch system prompt from zino-sys (`get`)
2. Fetch context from zino-ctx (`build`)
3. Assemble messages array: `[system?, ...context, user]`
4. Dispatch to zino-agr (`run` with `stream` flag); if agr is unavailable, fall back to direct zino-rtr dispatch (`infer`)
5. Store user+assistant exchange in zino-mem (`append_history`) if `channel_id` was provided

### Configuration

| Key | Default | Description |
|---|---|---|
| `[daemon] socket` | `/run/zino/daemon.sock` | UDS listen path |
| `[agent] temperature` | `0.7` | LLM temperature for inference |
| `[agent] top_p` | `1.0` | LLM top_p for inference |
| `[agr] max_iterations` | `5` | Max agentic loop iterations |

---

## zino-rtr

**File:** `zino-rtr.py`
**Socket:** `/run/zino/rtr.sock`
**Role:** LLM router. Maintains the API connection and performs inference requests, either streaming or collected.

### Messages

#### `capabilities`

```
→ {"type": "capabilities"}
← {"streaming": bool, "model": str}
```

Returns whether streaming is enabled and the configured model name.

#### `infer`

```
→ {"type": "infer", "messages": [...], "temperature": float, "top_p": float, "stream": bool}
```

| Field | Required | Description |
|---|---|---|
| `messages` | yes | OpenAI-format messages array |
| `temperature` | yes | Sampling temperature |
| `top_p` | yes | Nucleus sampling parameter |
| `stream` | yes | If `true` and streaming is capable, stream the response |

**Streaming response** (`stream: true` and streaming capable):
```
N x {"type": "chunk", "delta": str}
1 x {"type": "done", "full_text": str}
```

**Collected response** (otherwise):
```
1 x {"type": "response", "content": str}
```

**Error**:
```
1 x {"type": "error", "message": str}
```

### Configuration

| Key | Default | Description |
|---|---|---|
| `[rtr] socket` | `/run/zino/rtr.sock` | UDS listen path |
| `[rtr] streaming` | `true` | Enable streaming support |
| `[api] base_url` | — | OpenAI-compatible API endpoint |
| `[api] key` | — | API key (falls back to `LLM_API_KEY` env var) |
| `[service] model` | — | Model identifier |

---

## zino-agr

**File:** `zino-agr.py`
**Socket:** `/run/zino/agr.sock`
**Role:** Agentic runtime. Runs the inference loop: send prompt to rtr, parse response for interrupt tokens (`<tool_call>`, `<task>`), execute them, inject results, and repeat until no interrupts remain or max iterations is reached.

### Messages

#### `run`

```
→ {"type": "run", "messages": [...], "temperature": float, "top_p": float,
   "max_iterations": int?, "stream": bool?}
```

| Field | Required | Description |
|---|---|---|
| `messages` | yes | Fully assembled prompt (system + context + user) |
| `temperature` | yes | Sampling temperature |
| `top_p` | yes | Nucleus sampling parameter |
| `max_iterations` | no | Override default max iterations |
| `stream` | no | If `true`, stream chunks to the caller in real-time. Default `false` |

**Streaming response** (`stream: true`):
```
N x {"type": "chunk", "delta": str}
1 x {"type": "done", "full_text": str, "iterations": int}
```

In streaming mode, text tokens are forwarded as chunks in real-time. When an interrupt block's closing tag arrives, execution fires immediately and the result is sent as a chunk before streaming continues. If interrupts were found, the loop continues with a new iteration.

**Collected response** (`stream: false`):
```
1 x {"type": "result", "content": str, "iterations": int}
```

**Error** (either mode):
```
1 x {"type": "error", "message": str}
```

### Interrupt tokens

The LLM output is parsed for two types of interrupt blocks:

**Tool call:**
```xml
<tool_call>
{"name": "execute_bash", "arguments": {"code": "ls -la"}}
</tool_call>
```

Executed via zino-exc. The tool function name (e.g. `execute_bash`) is mapped to an executor name (e.g. `bash`) using definitions from the `tools/` directory.

**Task delegation:**
```xml
<task>
{"skill": "skill_name", "request": "what to do"}
</task>
```

Runs the skill pipeline: interpreter (understand request) → fabricator (produce code) → execute via zino-exc. Skill definitions are loaded from the `skills/` directory.

After each interrupt is executed, the result is injected into the response text as `<tool_response>...</tool_response>` or `<task_response>...</task_response>`, and the augmented text becomes the assistant message for the next iteration.

### StreamParser (internal)

Used in streaming mode to detect interrupt blocks character-by-character as tokens arrive. Three states:

- **FORWARDING** — normal text passes through; `<` triggers holdback
- **HOLDBACK** — buffering characters that might form `<tool_call>` or `<task>`; flushes as text if no tag matches
- **CAPTURING** — accumulating content inside a matched tag until the closing tag arrives

### Configuration

| Key | Default | Description |
|---|---|---|
| `[agr] socket` | `/run/zino/agr.sock` | UDS listen path |
| `[agr] max_iterations` | `5` | Default max agentic loop iterations |
| `[rtr] socket` | `/run/zino/rtr.sock` | Socket for rtr communication |
| `[exc] socket` | `/run/zino/exc.sock` | Socket for exc communication |
| `[tools] dir` | `tools` | Directory containing tool definitions |
| `[skills] dir` | `skills` | Directory containing skill definitions |

---

## zino-sys

**File:** `zino-sys.py`
**Socket:** `/run/zino/sys.sock`
**Role:** System prompt service. Loads a template (`SYSTEM.md`), substitutes placeholders with tool/skill definitions and runtime state, and serves the built prompt.

### Messages

#### `get`

```
→ {"type": "get"}
← {"type": "system_prompt", "content": str}
```

Returns the current built system prompt.

#### `reload`

```
→ {"type": "reload"}
← {"type": "system_prompt", "content": str}
```

Reloads tools and skills from disk, re-renders the template, and returns the new prompt.

#### `set_soft_memory`

```
→ {"type": "set_soft_memory", "content": str}
← {"type": "system_prompt", "content": str}
```

Replaces the `%%SOFT_MEMORIES%%` placeholder value and re-renders the template. Returns the new prompt.

### Template placeholders

| Placeholder | Source |
|---|---|
| `%%PERSONALITY%%` | `[sys] personality` in config |
| `%%TOOL_LIST%%` | Built from tool JSON files in `[tools] dir` |
| `%%SKILL_LINES%%` | Built from skill JSON files in `[skills] dir` |
| `%%MAX_ITERATIONS%%` | `[agent] max_iterations` in config |
| `%%SOFT_MEMORIES%%` | Runtime-mutable via `set_soft_memory` |

### Configuration

| Key | Default | Description |
|---|---|---|
| `[sys] socket` | `/run/zino/sys.sock` | UDS listen path |
| `[sys] template` | `SYSTEM.md` | Path to the system prompt template |
| `[sys] personality` | `""` | Personality text for `%%PERSONALITY%%` |
| `[sys] soft_memory` | `""` | Initial soft memory content |
| `[tools] dir` | `tools` | Directory containing tool definitions |
| `[skills] dir` | `skills` | Directory containing skill definitions |

---

## zino-exc

**File:** `zino-exc.py`
**Socket:** `/run/zino/exc.sock`
**Role:** Code execution service. Runs code produced by tools and skills, enforces timeouts, and optionally runs static analysis before execution.

### Messages

#### `execute`

```
→ {"type": "execute", "executor": str, "code": str}
← {"type": "result", "exit_code": int, "stdout": str, "stderr": str}
```

Writes `code` to a temporary file and executes it with the named executor (e.g. `bash`, `python3`). If the executor has a static analysis tool configured (e.g. `shellcheck`), validation runs first as a warning (does not block execution). Returns stdout, stderr, and exit code.

#### `validate`

```
→ {"type": "validate", "executor": str, "code": str}
← {"type": "validation", "ok": bool, "errors": str}
```

Runs only static analysis on the code without executing it. Returns `ok: true` if validation passes or no validator is configured.

#### `capabilities`

```
→ {"type": "capabilities"}
← {"type": "capabilities", "executors": {name: {...}}, "timeout": int}
```

Returns the registered executors and their metadata (command, file extension, static analysis tool, whether the validator is available) and the configured timeout.

### Configuration

| Key | Default | Description |
|---|---|---|
| `[exc] socket` | `/run/zino/exc.sock` | UDS listen path |
| `[exc] timeout` | `30` | Execution timeout in seconds |
| `[tools] dir` | `tools` | Directory containing tool definitions (used to discover executors) |

---

## zino-mem

**File:** `zino-mem.py`
**Socket:** `/run/zino/mem.sock`
**Role:** Memory service. Stores channel-based chat histories, hard memories with TF-IDF similarity search, and a mutable soft memory string. All data persists to disk.

### Messages — Chat history

#### `get_history`

```
→ {"type": "get_history", "channel_id": str, "limit": int?}
← {"type": "history", "messages": [...]}
```

Returns the chat history for a channel. `limit` restricts to the N most recent messages (0 or omitted = all).

#### `append_history`

```
→ {"type": "append_history", "channel_id": str, "messages": [...]}
← {"type": "ok"}
```

Appends messages to a channel's history. Each message is `{"role": str, "content": str}`.

#### `clear_history`

```
→ {"type": "clear_history", "channel_id": str}
← {"type": "ok"}
```

Deletes all history for a channel.

### Messages — Hard memories

#### `search_hard`

```
→ {"type": "search_hard", "query": str, "top_k": int?}
← {"type": "hard_memories", "results": [{"messages": [...], "score": float}, ...]}
```

Searches stored hard memories using TF-IDF cosine similarity against the query. Returns the top-k results (default 3) with their similarity scores.

#### `store_hard`

```
→ {"type": "store_hard", "messages": [...], "tag": str?}
← {"type": "ok", "id": int}
```

Stores a new hard memory entry (a list of messages with an optional tag for searchability). Returns the entry's index.

### Messages — Soft memory

#### `get_soft`

```
→ {"type": "get_soft"}
← {"type": "soft_memory", "content": str}
```

Returns the current soft memory string.

#### `set_soft`

```
→ {"type": "set_soft", "content": str}
← {"type": "ok"}
```

Replaces the soft memory string and persists to disk.

### Configuration

| Key | Default | Description |
|---|---|---|
| `[mem] socket` | `/run/zino/mem.sock` | UDS listen path |
| `[mem] data_dir` | `data/mem` | Directory for persistent storage |

### Storage layout

```
data/mem/
  channels/
    {channel_id}.json    # per-channel chat history
  hard_memories.json     # all hard memory entries
  soft_memory.txt        # soft memory string
```

---

## zino-ctx

**File:** `zino-ctx.py`
**Socket:** `/run/zino/ctx.sock`
**Role:** Context assembly service. Builds the context messages inserted between the system prompt and user message. Combines skill examples, hard memories (via zino-mem), and chat history (via zino-mem).

### Messages

#### `build`

```
→ {"type": "build", "channel_id": str?, "user_message": str}
← {"type": "context", "messages": [...]}
```

| Field | Required | Description |
|---|---|---|
| `channel_id` | no | If provided, chat history is included |
| `user_message` | yes | Used as the query for hard memory similarity search |

Returns a messages array assembled in this order:

1. **Skill examples** — user/assistant pairs from skill fabricator definitions (static, loaded at startup)
2. **Hard memories** — similarity search results from zino-mem using `user_message` as query
3. **Chat history** — recent messages from zino-mem for the given `channel_id`

### Configuration

| Key | Default | Description |
|---|---|---|
| `[ctx] socket` | `/run/zino/ctx.sock` | UDS listen path |
| `[ctx] history_limit` | `50` | Max chat history messages to include |
| `[ctx] hard_memory_top_k` | `3` | Number of hard memory results per search |
| `[mem] socket` | `/run/zino/mem.sock` | Socket for zino-mem communication |
| `[skills] dir` | `skills` | Directory containing skill definitions |

---

## zino_common

**File:** `zino_common.py`
**Role:** Shared library. Not a service — imported by all components.

### Wire protocol

All inter-service communication uses 4-byte big-endian length-prefixed JSON over Unix domain sockets.

#### `send_msg(writer, payload)`

Encodes `payload` as JSON, prepends a 4-byte length header, and writes to the stream.

#### `recv_msg(reader)`

Reads the 4-byte length header, then reads exactly that many bytes, and decodes as JSON.

### Utilities

#### `open_uds(path)`

Opens a Unix domain socket connection. Returns `(reader, writer)`.

#### `setup_logging(name, config)`

Configures and returns a logger. Log level is determined by (highest priority first):
1. `ZINO_LOG_LEVEL` environment variable
2. `[logging] level` in config
3. `"INFO"` default

---

## zino-cli

**File:** `zino-cli.py`
**Role:** Minimal test client. Not a service — connects to zino-daemon.

### Usage

```
python3 zino-cli.py "your message"
python3 zino-cli.py --stream "your message"
python3 zino-cli.py --channel my-channel --stream "your message"
python3 zino-cli.py --config /path/to/ZINO.toml "your message"
```

| Flag | Description |
|---|---|
| `--stream`, `-s` | Enable streaming mode (chunks print as they arrive) |
| `--channel`, `-ch` | Channel ID for history persistence |
| `--config`, `-c` | Config file path (default: `ZINO.toml` or `$ZINO_CONFIG`) |
