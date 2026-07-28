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
from httpx import ASGITransport, AsyncClient

from jupyter_mcp_server.assistant_api import app, runtime

EXPECTED_OPERATION_IDS = {
    "health",
    "list_files",
    "read_file",
    "write_library_file",
    "list_kernels",
    "list_notebooks",
    "use_notebook",
    "read_notebook",
    "get_notebook_status",
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
    "write_library_file",
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


@pytest.mark.asyncio
async def test_unexpected_route_failure_is_contained(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unexpected route crash must remain a wire-200 GPT error envelope."""

    def crash(_notebook_id: str) -> tuple[str, dict[str, object]]:
        raise RuntimeError("unexpected execution-status failure")

    monkeypatch.setattr(runtime, "execution_status", crash)
    transport = ASGITransport(app=app, raise_app_exceptions=True)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/notebooks/nb_test/execution")

    body = response.json()
    assert response.status_code == 200
    assert body["ok"] is False
    assert body["http_status"] == 500
    assert body["error_type"] == "RuntimeError"
    assert "unexpected execution-status failure" in body["error_message"]
    request_id = response.headers["X-Request-ID"]
    assert body["request_id"] == request_id
    assert any(
        "assistant_request_end" in record.message
        and f"request_id={request_id}" in record.message
        and "notebook_id=nb_test" in record.message
        for record in caplog.records
    )


def test_openapi_document_generates() -> None:
    """The OpenAPI document is generatable without error."""
    schema = app.openapi()
    assert schema["openapi"].startswith("3.")
    assert schema["info"]["title"] == "Jupyter Assistant API"
    assert schema.get("servers", []) == []


def test_restart_notebook_request_body_schema_is_tool_importable_object() -> None:
    """Tool importers require request body schemas to be concrete objects."""
    operation = app.openapi()["paths"]["/v1/notebooks/{notebook_id}/restart"]["post"]
    schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert schema == {"$ref": "#/components/schemas/RestartNotebookRequest"}


def test_read_notebook_schema_directs_bounded_overview_then_detail() -> None:
    """The Action schema must make the existing bounded workflow discoverable."""
    operation = app.openapi()["paths"]["/v1/notebooks/{notebook_id}"]["get"]
    parameters = {
        parameter["name"]: parameter
        for parameter in operation["parameters"]
    }

    format_description = parameters["response_format"]["description"]
    limit_description = parameters["limit"]["description"]
    assert "brief" in format_description
    assert "bounded 'detailed' page" in format_description
    assert "Action response limit" in limit_description
    assert parameters["limit"]["schema"]["maximum"] == 200


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
