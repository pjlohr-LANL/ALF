from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def task_accepts_argument(func: Callable[..., Any], argument_name: str) -> bool:
    try:
        return argument_name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def pyseqm_batch_size_from_qm_config(qm_config: dict[str, Any] | None, qm_task_func: Callable[..., Any]) -> int:
    requested_batch_size = max(1, int(dict(qm_config or {}).get("pyseqm_batch_size", 1)))
    if not task_accepts_argument(qm_task_func, "molecule_objects"):
        return 1
    return requested_batch_size
