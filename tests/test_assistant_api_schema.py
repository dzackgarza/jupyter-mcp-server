# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Unit tests for the Assistant API OpenAPI schema.

Covers spec required unit tests 5 and 6:

5. The generated OpenAPI document contains all expected ``operationId``
   values.
6. Mutation endpoints contain ``x-openai-isConsequential: false``.
"""

from __future__ import annotations

import pytest
from fastapi.routing import APIRoute

from jupyter_mcp_server.assistant_api import app

EXPECTED_OPERATION_IDS = {
    "health",
    "list_files",
    "list_kernels",
    "list_notebooks",
    "use_notebook",
    "read_notebook",
    "read_cell",
    "insert_cell",
    "overwrite_cell_source",
    "edit_cell_source",
    "delete_cell",
    "move_cell",
    "clear_cell_output",
    "execute_cell",
    "insert_execute_code_cell",
    "execute_code",
    "get_execution_status",
    "restart_notebook",
    "unuse_notebook",
}

MUTATION_OPERATION_IDS = {
    "use_notebook",
    "insert_cell",
    "overwrite_cell_source",
    "edit_cell_source",
    "delete_cell",
    "move_cell",
    "clear_cell_output",
    "execute_cell",
    "insert_execute_code_cell",
    "execute_code",
    "restart_notebook",
    "unuse_notebook",
}


def _api_routes() -> list[APIRoute]:
    return [r for r in app.routes if isinstance(r, APIRoute)]


def test_openapi_document_generates() -> None:
    """The OpenAPI document is generatable without error."""
    schema = app.openapi()
    assert schema["openapi"].startswith("3.")
    assert schema["info"]["title"] == "Jupyter Assistant API"


def test_all_expected_operation_ids_present() -> None:
    """Every expected operationId is present in the route table."""
    ops = {r.operation_id for r in _api_routes() if r.operation_id}
    missing = EXPECTED_OPERATION_IDS - ops
    assert not missing, f"Missing operationIds: {missing}"


def test_no_unexpected_operation_ids() -> None:
    """No extra operationIds beyond the spec."""
    ops = {r.operation_id for r in _api_routes() if r.operation_id}
    extra = ops - EXPECTED_OPERATION_IDS
    assert not extra, f"Unexpected operationIds: {extra}"


@pytest.mark.parametrize("op_id", sorted(MUTATION_OPERATION_IDS))
def test_mutation_endpoints_are_non_consequential(op_id: str) -> None:
    """Every mutation endpoint has x-openai-isConsequential: false.

    Without this, GPT Actions defaults non-GET operations to
    consequential and requires confirmation on every call.
    """
    route = next(r for r in _api_routes() if r.operation_id == op_id)
    extra = getattr(route, "openapi_extra", {}) or {}
    assert extra.get("x-openai-isConsequential") is False, (
        f"{op_id} missing x-openai-isConsequential: false"
    )


def test_get_endpoints_are_read_only() -> None:
    """Read endpoints use GET method."""
    read_ops = EXPECTED_OPERATION_IDS - MUTATION_OPERATION_IDS
    for op_id in read_ops:
        route = next(r for r in _api_routes() if r.operation_id == op_id)
        assert "GET" in route.methods, f"{op_id} is not GET: {route.methods}"
