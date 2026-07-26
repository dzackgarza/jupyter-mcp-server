# japi

`japi` exposes the deployed Jupyter Assistant API as commands generated from its live
OpenAPI document.

From the repository root:

```bash
./japi --help
./japi health
./japi get-notebook-status <notebook-id>
./japi read-notebook <notebook-id> --help
```

## Run the latest GitHub push

Run the launcher directly from the current `main` branch:

```bash
GONOSUMDB=github.com/dzackgarza/jupyter-mcp-server/tools/japi \
GOPROXY=direct \
go run github.com/dzackgarza/jupyter-mcp-server/tools/japi@main health
```

Replace `health` with any command shown by `--help`. This resolves the launcher from
GitHub on each invocation; the launcher then discovers its commands from the deployed
OpenAPI document. The scoped `GONOSUMDB` and `GOPROXY=direct` settings bypass a public
Go checksum-service error for the nested module without disabling checksum verification
for other modules.

For a globally callable `japi` that retains those semantics, put this shim at
`~/.local/bin/japi`:

```bash
#!/usr/bin/env bash
set -euo pipefail

export GONOSUMDB=github.com/dzackgarza/jupyter-mcp-server/tools/japi
export GOPROXY=direct
exec go run github.com/dzackgarza/jupyter-mcp-server/tools/japi@main "$@"
```

Ensure `~/.local/bin` is on `PATH`, then call `japi health` from any directory. Go
reuses its build and module caches after the first invocation.

Each invocation uses isolated temporary Restish state. It fetches
`https://jupyter-assistant.dzackgarza.com/openapi.json` before constructing the command
tree, so a stale cached schema cannot define a command.

Output is JSON. When the API returns its HTTP-200 error envelope, `japi` prints the
envelope and exits nonzero.

Restish v2.3.0 and its Go dependencies are checked into `vendor/`. The launcher uses
that tree without downloading dependencies. To build a standalone binary:

```bash
cd tools/japi
go build -mod=vendor -o japi .
```
