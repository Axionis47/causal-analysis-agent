"""Tracing helpers for LangSmith."""

from __future__ import annotations

import inspect
from typing import Any, Callable

try:
    from langsmith import traceable
except Exception:  # noqa: BLE001
    traceable = None


def traced(
    name: str,
    run_type: str = "tool",
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            metadata = {}
            if args and isinstance(args[0], dict):
                state = args[0]
                metadata = {
                    "analysis_id": state.get("analysis_id"),
                    "dataset_id": state.get("dataset_id"),
                    "kaggle_url": state.get("kaggle_url"),
                }
            elif args and isinstance(args[0], str):
                metadata = {"analysis_id": args[0]}
                if len(args) > 1 and isinstance(args[1], str):
                    metadata["kaggle_url"] = args[1]
            if traceable is None:
                return await func(*args, **kwargs)
            wrapped = traceable(name=name, run_type=run_type, metadata=metadata)(func)
            return await wrapped(*args, **kwargs)

        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            metadata = {}
            if args and isinstance(args[0], dict):
                state = args[0]
                metadata = {
                    "analysis_id": state.get("analysis_id"),
                    "dataset_id": state.get("dataset_id"),
                    "kaggle_url": state.get("kaggle_url"),
                }
            elif args and isinstance(args[0], str):
                metadata = {"analysis_id": args[0]}
                if len(args) > 1 and isinstance(args[1], str):
                    metadata["kaggle_url"] = args[1]
            if traceable is None:
                return func(*args, **kwargs)
            wrapped = traceable(name=name, run_type=run_type, metadata=metadata)(func)
            return wrapped(*args, **kwargs)

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
