import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lensieve.names import ImageViewField as IVF


class QueryField(StrEnum):
    PATH = IVF.PATH
    SHA256 = IVF.SHA256
    DATE_TAKEN = IVF.DATE_TAKEN
    CAMERA_MAKE = IVF.CAMERA_MAKE
    CAMERA_MODEL = IVF.CAMERA_MODEL
    IMAGE_FORMAT = IVF.IMAGE_FORMAT
    EXIF_ERROR = IVF.EXIF_ERROR
    STRUCTURE_ERROR = IVF.STRUCTURE_ERROR
    DISPLAY_WIDTH = IVF.DISPLAY_WIDTH
    DISPLAY_HEIGHT = IVF.DISPLAY_HEIGHT
    ORIENTATION_KIND = IVF.ORIENTATION_KIND
    ASPECT_RATIO = IVF.ASPECT_RATIO


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ColumnExpr(StrictBaseModel):
    column: QueryField


class AggregateOp(StrEnum):
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    COUNT_DUPLICATES = "count_duplicates"
    MIN = "min"
    MAX = "max"
    AVG = "avg"
    SUM = "sum"


class AggregateExpr(StrictBaseModel):
    op: AggregateOp
    column: QueryField | Literal["*"] = "*"


class DatePart(StrEnum):
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    DATE = "date"


class DatePartExpr(StrictBaseModel):
    part: DatePart
    column: Literal[QueryField.DATE_TAKEN] = QueryField.DATE_TAKEN


class SelectExpr(StrictBaseModel):
    expr: ColumnExpr | AggregateExpr | DatePartExpr
    alias: str


class FilterOp(StrEnum):
    EQ = "="
    NE = "!="

    LT = "<"
    LE = "<="

    GT = ">"
    GE = ">="

    IN = "in"
    NOT_IN = "not_in"

    BETWEEN = "between"

    CONTAINS = "contains"


LIST_FILTER_OPS = {FilterOp.IN, FilterOp.NOT_IN}
SIMPLE_FILTER_OPS = {
    FilterOp.EQ,
    FilterOp.NE,
    FilterOp.LT,
    FilterOp.LE,
    FilterOp.GT,
    FilterOp.GE,
}


class Filter(StrictBaseModel):
    column: QueryField
    op: FilterOp
    value: Any | None = None

    @model_validator(mode="after")
    def validate_value(self) -> "Filter":
        if self.value is None:
            if self.op in {FilterOp.EQ, FilterOp.NE}:
                return self
            raise ValueError(f"{self.op!r} cannot be used with null")

        if self.op in LIST_FILTER_OPS:
            if not isinstance(self.value, list | tuple) or len(self.value) == 0:
                raise ValueError(f"{self.op!r} requires a non-empty list")
            return self

        if self.op == FilterOp.BETWEEN:
            if not isinstance(self.value, list | tuple) or len(self.value) != 2:
                raise ValueError(f"{self.op!r} requires exactly two values")
            return self

        return self


class OrderDirection(StrEnum):
    ASC = "asc"
    DESC = "desc"


class OrderBy(StrictBaseModel):
    expr: str
    direction: OrderDirection = OrderDirection.ASC


_SAFE_ALIAS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class QuerySpec(StrictBaseModel):
    select: list[SelectExpr] = Field(min_length=1)
    filters: list[Filter] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    order_by: list[OrderBy] = Field(default_factory=list)
    limit: int = Field(default=100, ge=1, le=1000)

    @model_validator(mode="after")
    def validate_alias_references(self) -> "QuerySpec":
        invalid_aliases = [item.alias for item in self.select if not _SAFE_ALIAS_RE.fullmatch(item.alias)]
        if invalid_aliases:
            raise ValueError(f"invalid aliases: {invalid_aliases}")

        aliases = {item.alias for item in self.select}

        unknown_group_by = set(self.group_by) - aliases
        if unknown_group_by:
            raise ValueError(f"group_by references unknown aliases: {sorted(unknown_group_by)}")

        unknown_order_by = {item.expr for item in self.order_by} - aliases
        if unknown_order_by:
            raise ValueError(f"order_by references unknown aliases: {sorted(unknown_order_by)}")

        return self


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    sql: str | None = None
    params: list[Any] | None = None
