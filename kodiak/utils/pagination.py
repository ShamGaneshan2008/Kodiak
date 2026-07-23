import math
from typing import TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)

    @property
    def offset(self) -> int:
        return calculate_offset(self.page, self.page_size)


class PaginatedResult[T](BaseModel):
    items: list[T] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1)
    total_pages: int = Field(default=0, ge=0)


def calculate_offset(page: int, page_size: int) -> int:
    """Calculate the database offset based on page and page size."""
    if page < 1:
        raise ValueError("Page must be 1 or greater")
    if page_size < 1:
        raise ValueError("Page size must be 1 or greater")
    return (page - 1) * page_size


def paginate(
    items: list[T],
    total: int,
    params: PaginationParams,
) -> PaginatedResult[T]:
    """Create a paginated result set."""
    total_pages = math.ceil(total / params.page_size) if params.page_size > 0 else 0
    return PaginatedResult(
        items=items,
        total=total,
        page=params.page,
        page_size=params.page_size,
        total_pages=total_pages,
    )
