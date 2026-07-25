<!--
  ~ Copyright (c) 2024-2025 Datalayer, Inc. / dzackgarza
  ~
  ~ BSD 3-Clause License
  -->

# Jupyter Assistant API

**Stateless HTTP/OpenAPI adapter over Jupyter MCP Server tool classes, designed for GPT Actions.**

This is a thin FastAPI transport layer that replaces the MCP protocol with a stateless REST API. Each request names the target notebook via a deterministic `nb_<base64>` ID derived from its Jupyter-root-relative filepath — no session state, no persisted mappings, survives restarts.

---

## Architecture

```
Custom GPT → HTTPS → cloudflared tunnel → FastAPI adapter → existing tool classes → JupyterLab
```

- **Adapter**: Uvicorn on `127.0.0.1:4042`, `workers=1` (required by `asyncio.Lock` around `NotebookManager`)
- **Public URL**: `https://jupyter-assistant.dzackgarza.com`
- **JupyterLab**: port 8888, no auth, `root_dir=~/research/computations/notebooks`
- **Default kernelspec**: `sagemath` — kernel pre-started via Jupyter REST API (not the MCP tool's default `python3`)
- **Lock model**: single `asyncio.Lock` around all notebook-specific operations; one Uvicorn worker enforces serialization

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Deterministic `nb_<base64>` IDs | No persisted mapping; survives restart; reversible to filepath |
| `x-openai-isConsequential: false` on all 11 mutation routes | Enables "always allow" mode in ChatGPT Actions |
| Pre-start kernel via REST API | `UseNotebookTool` connects to a pre-started kernel with the requested kernelspec instead of defaulting to `python3` |
| `workers=1` | `NotebookManager` is process-wide state; >1 worker races on the current-notebook pointer |
| Global exception handler | Catches unhandled exceptions with structured JSON + traceback so the GPT can diagnose failures |

---

## API Endpoints

All routes are served under `https://jupyter-assistant.dzackgarza.com`. The full OpenAPI spec is at `/openapi.json`.

### Server-level

| Method | Path | Operation ID | Description |
|---|---|---|---|
| GET | `/health` | `health` | Report API and Jupyter server readiness |
| GET | `/v1/files` | `list_files` | List files on the Jupyter server |
| GET | `/v1/kernels` | `list_kernels` | List active kernels |
| GET | `/v1/notebooks` | `list_notebooks` | List all `.ipynb` files |

### Notebook lifecycle

| Method | Path | Operation ID | Description |
|---|---|---|---|
| POST | `/v1/notebooks/use` | `use_notebook` | Open or create a notebook (requires `notebook_path`, optional `kernel_name` defaults to `sagemath`) |
| POST | `/v1/notebooks/{notebook_id}/restart` | `restart_notebook` | Restart the notebook's kernel |

### Reading

| Method | Path | Operation ID | Description |
|---|---|---|---|
| GET | `/v1/notebooks/{notebook_id}` | `read_notebook` | Read notebook contents (paginated; `brief` or `detailed` format) |
| GET | `/v1/notebooks/{notebook_id}/cells/{cell_index}` | `read_cell` | Read a single cell by index |

### Cell mutations

| Method | Path | Operation ID | Description |
|---|---|---|---|
| POST | `/v1/notebooks/{notebook_id}/cells` | `insert_cell` | Insert a cell at the given index |
| PUT | `/v1/notebooks/{notebook_id}/cells/{cell_index}` | `overwrite_cell_source` | Overwrite cell source |
| PATCH | `/v1/notebooks/{notebook_id}/cells/{cell_index}` | `edit_cell_source` | Find-and-replace edit on cell source |
| DELETE | `/v1/notebooks/{notebook_id}/cells` | `delete_cell` | Delete one or more cells |
| POST | `/v1/notebooks/{notebook_id}/cells/move` | `move_cell` | Move a cell from source to target index |
| POST | `/v1/notebooks/{notebook_id}/cells/{cell_index}/clear-output` | `clear_cell_output` | Clear cell output |

### Execution

| Method | Path | Operation ID | Description |
|---|---|---|---|
| POST | `/v1/notebooks/{notebook_id}/cells/{cell_index}/execute` | `execute_cell` | Execute a cell by index |
| POST | `/v1/notebooks/{notebook_id}/cells/insert-and-execute` | `insert_execute_code_cell` | Insert + execute a code cell in one step |
| POST | `/v1/notebooks/{notebook_id}/execute-code` | `execute_code` | Execute temporary code without inserting a cell |

---

## GPT Actions Setup

This adapter is designed for ChatGPT Custom GPTs via the Actions feature.

### 1. Import the OpenAPI spec

1. Open the GPT editor → **Actions** → **Create new action**
2. Click **Import from URL**
3. Paste:
   ```
   https://jupyter-assistant.dzackgarza.com/openapi.json
   ```
4. The schema loads with all 17 endpoints, request/response models, and operation IDs

### 2. Authentication

Set to **None** (the adapter has no auth layer — it sits behind cloudflared on localhost).

If you add auth later, set the GPT to send an API key header and add a middleware check in `assistant_api.py`.

### 3. "Always allow" mutations

All 11 mutation endpoints carry `x-openai-isConsequential: false` in the OpenAPI spec. This tells ChatGPT the operations are non-destructive, enabling the **"Always allow"** toggle so the GPT doesn't prompt for confirmation on every write.

Without this, ChatGPT asks "Allow this action?" on every cell insert, execute, delete, etc. — unusable for a multi-step workflow.

### 4. Suggested system prompt

Add this to the GPT's **Instructions** field to teach it the notebook ID workflow:

```
You have access to a Jupyter notebook server via the Jupyter Assistant API.

Notebook workflow:
- List available notebooks with GET /v1/notebooks (returns *.ipynb files with paths)
- Open a notebook with POST /v1/notebooks/use { "notebook_path": "<path>" }
  - This returns a deterministic notebook_id (nb_<base64>) for all subsequent calls
  - Default kernel is sagemath; override with "kernel_name" parameter
- All cell operations use the notebook_id in the URL path
- Execute code with POST /v1/notebooks/{notebook_id}/execute-code for ephemeral snippets
- Insert persistent cells with POST /v1/notebooks/{notebook_id}/cells/insert-and-execute

Available kernels: sagemath (default), python3, pari_jupyter, gap, singular, lean4, octave, coconut, julia-1.10

Rules:
- Always list notebooks before opening one to confirm the path exists
- Read the notebook before editing to understand its structure
- Use brief format for read_notebook unless you need full output details
- Cell indices are 0-based; -1 means append
```

### 5. Test it

After saving the GPT, try:

> "List my notebooks and show me what's in periods/fermat-periods.ipynb"

The GPT should:
1. Call `list_notebooks` → get the file list
2. Call `use_notebook` with `periods/fermat-periods.ipynb` → get a `notebook_id`
3. Call `read_notebook` with that ID → return cell contents

---

## JupyterLab Setup

The adapter expects a running JupyterLab on port 8888 with no authentication and the `sagemath` kernelspec as default.

### Requirements

- **JupyterLab 4.4.x** with `jupyter-collaboration` (enables real-time model sync that the tool classes depend on)
- **A SageMath kernel** — or any other kernelspec you want as default
- **Notebooks root** — a directory where the adapter looks for `.ipynb` files (default: `~/research/computations/notebooks`)

### Quick start (manual)

```bash
jupyter lab \
  --port 8888 \
  --ip 0.0.0.0 \
  --no-browser \
  --IdentityProvider.token='' \
  --IdentityProvider.password_required=False \
  --ServerApp.disable_check_xsrf=True \
  --ServerApp.root_dir=~/research/computations/notebooks
```

Flags explained:
- `--IdentityProvider.token=''` and `--IdentityProvider.password_required=False` — no auth (the adapter runs on localhost; cloudflared handles HTTPS)
- `--ServerApp.disable_check_xsrf=True` — required for the adapter to make API calls without CSRF tokens
- `--ServerApp.root_dir` — the directory where notebook files live; this becomes the root for `list_notebooks` and `use_notebook` paths

### systemd user service (recommended)

Create `~/.config/systemd/user/jupyter-sagemath.service`:

```ini
[Unit]
Description=JupyterLab (SageMath kernel)
After=default.target

[Service]
Type=simple
ExecStartPre=/path/to/check-port-8888.py
ExecStart=/usr/bin/jupyter lab \
  --port 8888 \
  --ip 0.0.0.0 \
  --no-browser \
  --IdentityProvider.token='' \
  --IdentityProvider.password_required=False \
  --ServerApp.disable_check_xsrf=True \
  --ServerApp.root_dir=/home/YOU/research/computations/notebooks
Restart=on-failure
RestartSec=5
Environment=HOME=/home/YOU
Environment=JUPYTER_PORT=8888

[Install]
WantedBy=default.target
```

Then:
```bash
systemctl --user daemon-reload
systemctl --user enable --now jupyter-sagemath.service
systemctl --user status jupyter-sagemath.service
```

The `ExecStartPre` script (optional) kills any stale process already bound to port 8888 before starting. Remove the line if you don't need it.

### Verify

```bash
curl -s http://localhost:8888/api/status | python3 -m json.tool
# Should return Jupyter server status JSON

curl -s http://localhost:8888/api/kernelspecs | python3 -c "import sys,json; print(list(json.load(sys.stdin)['kernelspecs'].keys()))"
# Should list available kernelspecs, e.g. ['sagemath', 'python3', ...]
```

### Available kernels on this machine

| Kernelspec | Language | Notes |
|---|---|---|
| `sagemath` | SageMath | Default for the adapter |
| `python3` | Python 3 | Standard CPython |
| `pari_jupyter` | PARI/GP | Number theory |
| `gap` | GAP | Group theory |
| `singular` | Singular | Algebraic geometry |
| `lean4` | Lean 4 | Theorem prover |
| `octave` | GNU Octave | MATLAB-compatible |
| `coconut` | Coconut | Pattern matching |
| `julia-1.10` | Julia 1.10 | Scientific computing |

Pass any of these as `kernel_name` in the `use_notebook` request body.

---

## Setup

### Prerequisites

- Python 3.10+
- JupyterLab 4.4.x running locally on port 8888 (see JupyterLab Setup above)
- A SageMath kernel (`sagemath`) installed and available in Jupyter

### Install

```bash
git clone https://github.com/dzackgarza/jupyter-assistant-api
cd jupyter-assistant-api
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configure

Set environment variables:

```bash
export JUPYTER_URL=http://localhost:8888
export JUPYTER_TOKEN=           # empty if JupyterLab has no auth
export PORT=4042                 # default: 4042
```

### Run

```bash
# Via the console entry point (recommended):
jupyter-assistant-api

# Or directly with uvicorn:
uvicorn jupyter_mcp_server.assistant_api:app --host 127.0.0.1 --port 4042 --workers 1
```

### systemd user services (recommended)

Both long-running pieces — the adapter and the Cloudflare tunnel — are vendored
as systemd user units in [`dev/systemd/`](dev/systemd). Install them by absolute
path so systemd symlinks the repo copies and the repo stays the single source of
truth:

```bash
systemctl --user enable --now \
  "$PWD/dev/systemd/jupyter-assistant-api.service" \
  "$PWD/dev/systemd/jupyter-assistant-tunnel.service"

systemctl --user is-active jupyter-assistant-api jupyter-assistant-tunnel
```

After editing a vendored unit, `systemctl --user daemon-reload && systemctl --user restart <unit>`.

The units hardcode `/home/dzack` paths — this deployment is single-machine by
design. Adjust the paths when installing elsewhere. `jupyter-assistant-tunnel`
reads `~/.cloudflared/config-jupyter-assistant.yml`, which is **not** vendored
because it names a credentials file; its ingress must point at `127.0.0.1:4042`.
If that host:port disagrees with the adapter, the public hostname serves
Cloudflare **error 502**; if the tunnel is not running at all, **error 1033**.

### Verify

```bash
curl http://127.0.0.1:4042/health
# {"ok":true,"status":"healthy","jupyter_url":"http://localhost:8888"}
```

### Logging

The adapter runs as a single uvicorn worker. All logs go to **stdout/stderr** of the terminal (or systemd journal) where uvicorn was started.

#### Where to look

| Scenario | How to check logs |
|---|---|
| Running in a terminal | Logs print directly to that terminal |
| Running in a tmux session | `tmux attach -t <session>` and scroll |
| systemd user service | `journalctl --user -u jupyter-assistant-api -f` |
| Piped to file | Start with `jupyter-assistant-api 2>&1 \| tee adapter.log` |

#### What gets logged

**Uvicorn access log** (one line per request):
```
INFO:     127.0.0.1:54321 - "POST /v1/notebooks/use HTTP/1.1" 200 OK
INFO:     127.0.0.1:54321 - "POST /v1/notebooks/nb_cGVyaW9kcy9mZXJtYXQtcGVyaW9kcy5pcHluYg/execute-code HTTP/1.1" 200 OK
```

**Uvicorn error log** (unhandled exceptions — but the global exception handler catches most):
```
ERROR:    Exception in ASGI application
```

**Upstream tool logs** (kernel operations, notebook reads):
```
INFO:jupyter_mcp_server.tools:Executing cell 3 in notebook periods/fermat-periods.ipynb
INFO:jupyter_mcp_server.tools:Kernel status: busy -> idle
```

#### Error responses

Every error returns structured JSON with diagnostic context:

```json
{
  "ok": false,
  "error_type": "HTTPException",
  "error_message": "Failed to start kernel 'bad_kernel': 404 ...",
  "traceback": "Traceback (most recent call last):\n  ...",
  "notebook_id": "nb_cGVyaW9kcy9mZXJtYXQtcGVyaW9kcy5pcHluYg",
  "notebook_path": "periods/fermat-periods.ipynb"
}
```

The `traceback` field contains the full Python traceback — useful for debugging kernel connection failures or tool class errors without digging through server logs.

#### Log level

To increase verbosity, pass uvicorn's `--log-level` flag:

```bash
uvicorn jupyter_mcp_server.assistant_api:app --host 127.0.0.1 --port 4042 --workers 1 --log-level debug
```

For Python-level debug logging (tool classes, kernel client):

```bash
JUPYTER_MCP_LOG_LEVEL=debug jupyter-assistant-api
```

#### systemd service logging

If running as a systemd user service, append to the unit file:

```ini
[Service]
StandardOutput=journal
StandardError=journal
SyslogIdentifier=jupyter-assistant-api
```

Then:
```bash
# Follow logs in real time:
journalctl --user -u jupyter-assistant-api -f

# Show last 50 lines:
journalctl --user -u jupyter-assistant-api -n 50

# Show errors only:
journalctl --user -u jupyter-assistant-api -p err
```

---

## Notebook ID Scheme

A notebook path like `periods/fermat-periods.ipynb` becomes:

```
path:                  periods/fermat-periods.ipynb
base64(urlsafe):       cGVyaW9kcy9mZXJtYXQtcGVyaW9kcy5pcHluYg==
stripped padding:      cGVyaW9kcy9mZXJtYXQtcGVyaW9kcy5pcHluYg
notebook_id:           nb_cGVyaW9kcy9mZXJtYXQtcGVyaW9kcy5pcHluYg
```

The ID is **not** a secret — only a route-safe representation of the filepath. Decode it with `decode_notebook_id()` from `jupyter_mcp_server.notebook_id`.

---

## Testing

```bash
pytest tests/test_notebook_id.py tests/test_assistant_api_schema.py -v
```

Integration tests require a running JupyterLab on port 8888:

```bash
pytest tests/test_assistant_api_integration.py -v
```

---

## cloudflared Tunnel

The public endpoint at `https://jupyter-assistant.dzackgarza.com` is served
through a Cloudflare named tunnel that routes HTTPS traffic to the local
adapter at `127.0.0.1:4042`.

### 1. Install cloudflared

```bash
# Debian/Ubuntu
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb
sudo dpkg -i /tmp/cloudflared.deb

# macOS
brew install cloudflared
```

Verify:
```bash
cloudflared --version
# cloudflared version 2026.7.2 (built 20260716-05:07:11)
```

### 2. Authenticate with Cloudflare

```bash
cloudflared tunnel login
# Opens a browser; pick the zone for dzackgarza.com
# Writes ~/.cloudflared/cert.pem
```

### 3. Create the named tunnel

```bash
cloudflared tunnel create jupyter-assistant
# Outputs tunnel ID: f12f0463-0152-489d-a028-cb7ae016f188
# Writes ~/.cloudflared/f12f0463-0152-489d-a028-cb7ae016f188.json
```

### 4. Map the hostname

```bash
cloudflared tunnel route dns jupyter-assistant jupyter-assistant.dzackgarza.com
```

### 5. Create the config file

Write `~/.cloudflared/config-jupyter-assistant.yml`:

```yaml
tunnel: f12f0463-0152-489d-a028-cb7ae016f188
credentials-file: /home/dzack/.cloudflared/f12f0463-0152-489d-a028-cb7ae016f188.json

ingress:
  - hostname: jupyter-assistant.dzackgarza.com
    service: http://127.0.0.1:4042
  - service: http_status:404
```

The last rule is a catch-all that returns 404 for unmatched hostnames — required by cloudflared.

### 6. Run the tunnel

```bash
# Foreground (for testing):
cloudflared tunnel --config ~/.cloudflared/config-jupyter-assistant.yml run

# Background (detached):
nohup cloudflared tunnel --config ~/.cloudflared/config-jupyter-assistant.yml run &
```

### 7. (Optional) Set up as a systemd user service

Create `~/.config/systemd/user/cloudflared-jupyter-assistant.service`:

```ini
[Unit]
Description=Cloudflare Tunnel - Jupyter Assistant API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/cloudflared tunnel --config /home/dzack/.cloudflared/config-jupyter-assistant.yml run
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

Then:
```bash
systemctl --user daemon-reload
systemctl --user enable --now cloudflared-jupyter-assistant.service
systemctl --user status cloudflared-jupyter-assistant.service
```

### Verification

```bash
curl -s https://jupyter-assistant.dzackgarza.com/health | python3 -m json.tool
# {
#     "ok": true,
#     "status": "healthy",
#     "jupyter_url": "http://localhost:8888"
# }
```

---

## Upstream

This repo is a fork of [datalayer/jupyter-mcp-server](https://github.com/datalayer/jupyter-mcp-server). The MCP server infrastructure (tool classes, notebook manager, kernel client) is preserved; only the transport layer is replaced. Upstream documentation for tool behavior lives at [jupyter-mcp-server.datalayer.tech](https://jupyter-mcp-server.datalayer.tech).

## Error contract

Every failing request returns **HTTP 200** with `ok: false`. The status the
failure would otherwise have carried is in `http_status`; clients branch on
`ok`, never on the wire status.

```json
{
  "ok": false,
  "http_status": 404,
  "error_type": "ToolError",
  "error_message": "[list_files] ... 404: file or directory '/nope' does not exist",
  "traceback": "...",
  "notebook_id": null,
  "notebook_path": null
}
```

This is deliberate. The consumer is a GPT Action whose HTTP client calls
`raise_for_status()`, so on any non-2xx it raises and shows the caller only
the exception type (`ClientResponseError: <class 'aiohttp...'>`) — the
response body, and every diagnostic in it, is discarded before the model sees
it. Returning 200 is what makes failures readable to the only thing reading
them.

`http_status` carries the real cause: a Jupyter-boundary status is propagated
from the exception chain (404 for a missing path) rather than flattened to
500, so a caller can distinguish "wrong path" from "server fault".

### Sage kernel

`.envrc` prepends `dev/jupyter` to `JUPYTER_PATH`, which shadows the system
`sagemath` kernelspec with one whose `argv` uses `sage --python`. The system
spec uses a bare `python`, resolved from the launching process's PATH, so a
Jupyter server started under this project's `.venv` gets an interpreter that
cannot `import sage`; the kernel then crash-loops and the websocket handshake
fails with an opaque 500. Run `direnv allow` once after cloning.
