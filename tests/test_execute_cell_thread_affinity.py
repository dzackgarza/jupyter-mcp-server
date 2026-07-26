# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Regression proof for PyCRDT ownership during notebook cell execution."""

import asyncio
import gc
import time

import pytest
from jupyter_nbmodel_client import NotebookModel

from jupyter_mcp_server.utils import execute_cell_with_forced_sync


class BlockingKernel:
    """The kernel side is irrelevant to the YDoc ownership boundary."""

    def execute_interactive(self, _source, **_kwargs):
        time.sleep(2)
        return {"content": {"execution_count": 1, "status": "ok"}}

    def stop(self):
        pass


@pytest.mark.asyncio
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
async def test_cancelled_execution_does_not_drop_yobjects_on_worker_thread(
    capfd,
):
    """A timed-out request must not leave the worker as the YDoc's last owner."""

    async def execute_request():
        notebook = NotebookModel()
        notebook.add_code_cell("print('finished')")
        await execute_cell_with_forced_sync(
            notebook, 0, BlockingKernel(), timeout_seconds=0.05
        )

    with pytest.raises(asyncio.TimeoutError):
        await execute_request()

    gc.collect()
    await asyncio.sleep(2.25)
    gc.collect()

    stderr = capfd.readouterr().err
    assert "unsendable, but is being dropped on another thread" not in stderr
