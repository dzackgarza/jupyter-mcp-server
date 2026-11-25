# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Generate a development kernelspec using Sage's own Python interpreter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    try:
        interpreter = (
            os.environ.get("SAGE_PYTHON")
            or subprocess.check_output(
                ["sage", "-c", "import sys; print(sys.executable)"], text=True
            ).strip()
        )
        # Do not resolve symlinks: a venv's python may link to a system Python,
        # and invoking the resolved target would lose the venv.
        if not Path(interpreter).is_absolute():
            raise ValueError(f"Sage Python is not an absolute path: {interpreter!r}")
        subprocess.run(
            [interpreter, "-c", "import sage.repl.ipython_kernel"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(
            f"Cannot configure the Sage kernel: {exc}. "
            "Use a local Sage installation or set SAGE_PYTHON to its absolute "
            "Python executable in the Jupyter server's environment."
        )

    spec = json.loads((root / "dev/jupyter/sagemath-kernel.json").read_text())
    spec["argv"][0] = interpreter
    data_dir = root / ".jupyter/dev-kernels"
    kernel_dir = data_dir / "kernels/sagemath"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(data_dir)


if __name__ == "__main__":
    main()
