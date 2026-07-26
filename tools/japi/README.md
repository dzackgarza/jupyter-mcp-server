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
