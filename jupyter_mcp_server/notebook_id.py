# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Deterministic notebook identity for the Assistant API.

A notebook is identified by its Jupyter-root-relative filepath.  For use in
URL path segments the path is encoded as URL-safe base64 with an ``nb_``
prefix, yielding a deterministic, reversible identifier that survives
adapter restarts with no persisted mapping.

The ID is not a secret; it is only a route-safe representation of the
filepath.
"""

from __future__ import annotations

from base64 import urlsafe_b64decode, urlsafe_b64encode
from pathlib import PurePosixPath

__all__ = [
    "decode_notebook_id",
    "encode_notebook_id",
    "normalize_notebook_path",
]


def normalize_notebook_path(value: str) -> str:
    """Normalize a Jupyter-root-relative notebook path.

    Parameters
    ----------
    value:
        Raw path string; may use POSIX or Windows separators.

    Returns
    -------
    str
        Normalized POSIX-style relative path with no leading ``./``.

    Raises
    ------
    ValueError
        If the path is absolute, contains ``..``, or does not end in
        ``.ipynb``.
    """
    path = PurePosixPath(value.replace("\\", "/"))

    if path.is_absolute():
        raise ValueError("Notebook path must be relative to the Jupyter root")
    # Reject Windows drive letters (e.g. C:/...) which PurePosixPath on Linux
    # does not treat as absolute.
    if len(value) >= 2 and value[1] == ":" and value[0].isalpha():
        raise ValueError("Notebook path must be relative to the Jupyter root")
    if ".." in path.parts:
        raise ValueError("Notebook path cannot contain '..'")
    if path.suffix != ".ipynb":
        raise ValueError("Notebook path must end in .ipynb")

    normalized = path.as_posix()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def encode_notebook_id(path: str) -> str:
    """Encode a notebook path as a deterministic ``nb_<base64>`` identifier."""
    normalized = normalize_notebook_path(path)
    encoded = urlsafe_b64encode(normalized.encode()).decode().rstrip("=")
    return f"nb_{encoded}"


def decode_notebook_id(notebook_id: str) -> str:
    """Decode an ``nb_<base64>`` identifier back to a normalized path.

    Raises
    ------
    ValueError
        If the identifier is malformed or the decoded path is invalid.
    """
    if not notebook_id.startswith("nb_"):
        raise ValueError("Invalid notebook ID")

    encoded = notebook_id[3:]
    encoded += "=" * (-len(encoded) % 4)
    return normalize_notebook_path(urlsafe_b64decode(encoded).decode())
