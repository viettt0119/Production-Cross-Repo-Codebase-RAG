from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol


class Span(Protocol):
    def set_attribute(self, key: str, value: Any) -> None: ...

    def set_output(self, value: Any) -> None: ...

    def record_error(self, error: BaseException) -> None: ...


class Tracer(Protocol):
    def span(self, name: str, *, input: Any = None, **attributes: Any) -> AbstractContextManager[Span]: ...

    @property
    def current_trace_id(self) -> str | None: ...

    def flush(self) -> None: ...
