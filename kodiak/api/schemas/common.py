from __future__ import annotations

from collections.abc import Sequence
from math import ceil

from pydantic import BaseModel
from pydantic import ConfigDict

class PaginatedResponse[T](BaseModel):
    model_config = ConfigDict(from_attributes=True)

    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int

    @classmethod
    def build(
        cls,
        items: Sequence[T],
        total: int,
        page: int,
        page_size: int,
    ) -> PaginatedResponse[T]:
        return cls(
            items=list(items),
            total=total,
            page=page,
            page_size=page_size,
            total_pages=ceil(total / page_size) if page_size else 0,
        )
