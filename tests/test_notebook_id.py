# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Unit tests for deterministic notebook identity.

Covers the required unit tests from the spec:

1. ``normalize_notebook_path`` rejects absolute paths, ``..``, and
   non-notebook files.
2. Encoding and decoding a notebook path are inverse operations.
3. The same path always produces the same ID.
4. Different paths produce different IDs.
"""

from __future__ import annotations

import pytest

from jupyter_mcp_server.notebook_id import (
    decode_notebook_id,
    encode_notebook_id,
    normalize_notebook_path,
)

# ---------------------------------------------------------------------------
# 1. normalize_notebook_path validation
# ---------------------------------------------------------------------------


class TestNormalizeNotebookPath:
    @pytest.mark.parametrize(
        "bad_path",
        [
            "/absolute/path.ipynb",
            "/home/user/research/tau-invariants.ipynb",
            "C:\\Users\\foo\\notebook.ipynb",  # Windows drive letter
            "D:/abs/notebook.ipynb",  # Windows drive with forward slash
        ],
    )
    def test_rejects_absolute_paths(self, bad_path: str) -> None:
        with pytest.raises(ValueError, match="relative to the Jupyter root"):
            normalize_notebook_path(bad_path)

    @pytest.mark.parametrize(
        "bad_path",
        [
            "../escape.ipynb",
            "research/../escape.ipynb",
            "research/foo/../../escape.ipynb",
            "..",
        ],
    )
    def test_rejects_parent_traversal(self, bad_path: str) -> None:
        with pytest.raises(ValueError, match=r"\.\."):
            normalize_notebook_path(bad_path)

    @pytest.mark.parametrize(
        "bad_path",
        [
            "research/tau-invariants.py",
            "research/tau-invariants",
            "research/tau-invariants.txt",
            "research/tau-invariants.ipynb.bak",
            "no_extension",
        ],
    )
    def test_rejects_non_notebook_files(self, bad_path: str) -> None:
        with pytest.raises(ValueError, match=r"\.ipynb"):
            normalize_notebook_path(bad_path)

    def test_strips_leading_dot_slash(self) -> None:
        assert normalize_notebook_path("./research/foo.ipynb") == "research/foo.ipynb"

    def test_normalizes_windows_separators(self) -> None:
        assert normalize_notebook_path("research\\sub\\foo.ipynb") == "research/sub/foo.ipynb"

    def test_accepts_nested_relative(self) -> None:
        assert normalize_notebook_path("research/enriques/tau-invariants.ipynb") == (
            "research/enriques/tau-invariants.ipynb"
        )


# ---------------------------------------------------------------------------
# 2. Encode/decode inverse
# ---------------------------------------------------------------------------


class TestEncodeDecodeInverse:
    @pytest.mark.parametrize(
        "path",
        [
            "research/tau-invariants.ipynb",
            "research/enriques/tau-invariants.ipynb",
            "n.ipynb",
            "a/b/c/d/e/f/deep.ipynb",
            "research\\sub\\foo.ipynb",  # windows sep
            "./research/foo.ipynb",  # leading ./
        ],
    )
    def test_roundtrip(self, path: str) -> None:
        normalized = normalize_notebook_path(path)
        encoded = encode_notebook_id(path)
        decoded = decode_notebook_id(encoded)
        assert decoded == normalized

    def test_decode_rejects_missing_prefix(self) -> None:
        with pytest.raises(ValueError, match="Invalid notebook ID"):
            decode_notebook_id("cmVzZWFyY2gvdGF1LWludmFyaWFudHMuaXB5bmI")

    def test_decode_rejects_garbage(self) -> None:
        with pytest.raises(ValueError, match="Invalid notebook ID"):
            decode_notebook_id("nb_!!!not-base64!!!")

    def test_decode_rejects_non_ipynb_payload(self) -> None:
        # base64 of "foo.py" — valid base64 but fails path validation
        from base64 import urlsafe_b64encode

        encoded = urlsafe_b64encode(b"foo.py").decode().rstrip("=")
        with pytest.raises(ValueError, match=r"\.ipynb"):
            decode_notebook_id(f"nb_{encoded}")


# ---------------------------------------------------------------------------
# 3. Determinism: same path -> same ID
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_path_same_id(self) -> None:
        p = "research/enriques/tau-invariants.ipynb"
        assert encode_notebook_id(p) == encode_notebook_id(p)

    def test_leading_dot_slash_same_id(self) -> None:
        assert encode_notebook_id("./research/foo.ipynb") == encode_notebook_id(
            "research/foo.ipynb"
        )

    def test_windows_sep_same_id(self) -> None:
        assert encode_notebook_id("research\\sub\\foo.ipynb") == encode_notebook_id(
            "research/sub/foo.ipynb"
        )


# ---------------------------------------------------------------------------
# 4. Different paths -> different IDs
# ---------------------------------------------------------------------------


class TestUniqueness:
    def test_different_paths_different_ids(self) -> None:
        paths = [
            "research/tau-invariants.ipynb",
            "research/enriques/tau-invariants.ipynb",
            "research/enriques/tau-invariants-2.ipynb",
            "other/tau-invariants.ipynb",
            "research/tau-invariants.ipynb2.ipynb",
        ]
        ids = [encode_notebook_id(p) for p in paths]
        assert len(set(ids)) == len(paths), f"Collision in: {ids}"

    def test_id_has_nb_prefix(self) -> None:
        assert encode_notebook_id("foo.ipynb").startswith("nb_")

    def test_id_is_url_safe(self) -> None:
        """No ``/``, ``+``, or ``=`` in the encoded id (URL path safe)."""
        import re

        eid = encode_notebook_id("research/sub/deep path with spaces.ipynb")
        assert re.match(r"^nb_[A-Za-z0-9_-]+$", eid), f"unsafe chars in {eid!r}"


# ---------------------------------------------------------------------------
# Spec example
# ---------------------------------------------------------------------------


def test_spec_example_roundtrip() -> None:
    """The spec's canonical example encodes/decodes correctly."""
    path = "research/enriques/tau-invariants.ipynb"
    eid = encode_notebook_id(path)
    assert eid.startswith("nb_")
    assert decode_notebook_id(eid) == path
