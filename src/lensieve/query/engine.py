import logging
import re
from datetime import datetime, timedelta
from typing import Any, cast

from lensieve.data.data_store import DataStore
from lensieve.data.utils import duck_table_name, sql_ident
from lensieve.names import TableName as TN
from lensieve.query.spec import (
    LIST_FILTER_OPS,
    SIMPLE_FILTER_OPS,
    AggregateExpr,
    AggregateOp,
    ColumnExpr,
    DatePart,
    DatePartExpr,
    Filter,
    FilterOp,
    QueryField,
    QueryResult,
    QuerySpec,
)

logger = logging.getLogger(__name__)

_SAFE_ALIAS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_alias(alias: str) -> None:
    if not _SAFE_ALIAS_RE.fullmatch(alias):
        raise ValueError(f"Unsafe alias: {alias!r}")


def _select_expr_sql(expr: ColumnExpr | AggregateExpr | DatePartExpr) -> str:
    if isinstance(expr, ColumnExpr):
        return sql_ident(expr.column)

    if isinstance(expr, DatePartExpr):
        col = sql_ident(expr.column)

        if expr.part == DatePart.DATE:
            return f"CAST({col} AS DATE)"

        return f"DATE_PART('{expr.part}', {col})"

    if isinstance(expr, AggregateExpr):
        if expr.op == AggregateOp.COUNT:
            if expr.column == "*":
                return "COUNT(*)"
            return f"COUNT({sql_ident(expr.column)})"

        if expr.op == AggregateOp.COUNT_DISTINCT:
            if expr.column == "*":
                raise ValueError(f"{expr.op!r} does not support '*'")
            return f"COUNT(DISTINCT {sql_ident(expr.column)})"

        if expr.op == AggregateOp.COUNT_DUPLICATES:
            if expr.column == "*":
                raise ValueError(f"{expr.op!r} does not support '*'")
            col = sql_ident(expr.column)
            return f"(COUNT({col}) - COUNT(DISTINCT {col}))"

        if expr.column == "*":
            raise ValueError(f"{expr.op!r} does not support '*'")

        return f"{expr.op.upper()}({sql_ident(expr.column)})"

    raise TypeError(f"Unsupported select expression: {expr!r}")


def _is_date_only_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _next_day(value: str) -> str:
    d = datetime.strptime(value, "%Y-%m-%d").date()
    return (d + timedelta(days=1)).isoformat()


def _filter_sql(filter_: Filter, params: list[Any]) -> str:
    col = sql_ident(filter_.column)
    op = filter_.op

    value = filter_.value

    if value is None:
        if op == FilterOp.EQ:
            return f"{col} IS NULL"
        if op == FilterOp.NE:
            return f"{col} IS NOT NULL"
        raise ValueError(f"{op} cannot be used with null")

    if op in SIMPLE_FILTER_OPS:
        if filter_.column == QueryField.DATE_TAKEN and _is_date_only_string(value):
            # Interpret date-only bounds as whole calendar days.
            # date_taken <= 2025-12-31 means date_taken < 2026-01-01.
            # date_taken > 2025-12-31 means date_taken >= 2026-01-01.
            if op == FilterOp.LE:
                op = FilterOp.LT
                value = _next_day(value)
            elif op == FilterOp.GT:
                op = FilterOp.GE
                value = _next_day(value)

        params.append(value)
        return f"{col} {op.value} ?"

    if op in LIST_FILTER_OPS:
        values = list(cast(list[Any] | tuple[Any, ...], value))
        params.extend(values)

        placeholders = ", ".join("?" for _ in values)
        sql_op = "IN" if op == FilterOp.IN else "NOT IN"

        return f"{col} {sql_op} ({placeholders})"

    if op == FilterOp.BETWEEN:
        low, high = cast(list[Any] | tuple[Any, Any], value)

        if filter_.column == QueryField.DATE_TAKEN and _is_date_only_string(high):
            params.extend([low, _next_day(high)])
            return f"{col} >= ? AND {col} < ?"

        params.extend([low, high])
        return f"{col} BETWEEN ? AND ?"

    if op == FilterOp.CONTAINS:
        params.append(f"%{value}%")
        return f"{col} ILIKE ?"

    raise ValueError(f"Unsupported filter op: {op!r}")


def compile_query(spec: QuerySpec, table_name: str) -> tuple[str, list[Any]]:
    params: list[Any] = []

    for item in spec.select:
        _validate_alias(item.alias)

    select_parts = [f"{_select_expr_sql(item.expr)} AS {sql_ident(item.alias)}" for item in spec.select]

    sql_parts = [
        "SELECT",
        ", ".join(select_parts),
        f"FROM {duck_table_name(table_name)}",
    ]

    if spec.filters:
        where_parts = [_filter_sql(item, params) for item in spec.filters]
        sql_parts.append("WHERE " + " AND ".join(where_parts))

    if spec.group_by:
        group_parts = []

        # group_by references select aliases, not raw column names or expressions
        for alias in spec.group_by:
            select_item = next(x for x in spec.select if x.alias == alias)
            group_parts.append(_select_expr_sql(select_item.expr))

        sql_parts.append("GROUP BY " + ", ".join(group_parts))

    if spec.order_by:
        order_parts = [f"{sql_ident(item.expr)} {item.direction.upper()}" for item in spec.order_by]
        sql_parts.append("ORDER BY " + ", ".join(order_parts))

    sql_parts.append("LIMIT ?")
    params.append(spec.limit)

    query = "\n".join(sql_parts)
    logger.info("Compiled SQL query:\n%s\nParams: %s", query, params)
    return query, params


def run_query(
    data_store: DataStore,
    spec: QuerySpec,
    *,
    include_sql: bool = False,
) -> QueryResult:
    sql, params = compile_query(spec, TN.IMAGES_VIEW)

    with data_store.connect_duckdb_for_lance() as con:
        result = con.execute(sql, params)
        columns = [desc[0] for desc in result.description]
        rows = [dict(zip(columns, row, strict=True)) for row in result.fetchall()]

    res = QueryResult(
        columns=columns,
        rows=rows,
        row_count=len(rows),
        sql=sql if include_sql else None,
        params=params if include_sql else None,
    )

    return res
