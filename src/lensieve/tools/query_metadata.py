from typing import Any

import sqlglot
from pydantic import BaseModel, Field, ValidationError
from sqlglot import expressions as exp

from lensieve.data.data_store import DataStore
from lensieve.data.utils import DUCKDB_CATALOG, DUCKDB_SCHEMA
from lensieve.names import TableName as TN
from lensieve.tools.errors import ToolError


class SqlQueryArgs(BaseModel):
    sql: str = Field(min_length=1)


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    sql: str | None = None
    params: list[Any] | None = None


FORBIDDEN_SQL_TYPES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Copy,
    exp.Attach,
    exp.Detach,
)


FORBIDDEN_FUNCTIONS = frozenset(
    {
        "read_csv",
        "read_csv_auto",
        "read_parquet",
        "read_json",
        "read_text",
        "read_blob",
        "glob",
    }
)


ALLOWED_TABLES = frozenset({TN.IMAGES_VIEW})


def _qualify_allowed_tables(root: exp.Select) -> exp.Select:
    root = root.copy()

    for table in root.find_all(exp.Table):
        table_name = table.name.lower()

        if table_name not in ALLOWED_TABLES:
            raise ValueError(f"Unknown or disallowed table: {table_name!r}")

        table.set("catalog", exp.to_identifier(DUCKDB_CATALOG, quoted=True))
        table.set("db", exp.to_identifier(DUCKDB_SCHEMA, quoted=True))
        table.set("this", exp.to_identifier(table_name, quoted=True))

    return root


def _enforce_limit(root: exp.Select, max_results: int) -> exp.Select:
    root = root.copy()

    limit = root.args.get("limit")

    if limit is None:
        root.set("limit", exp.Limit(expression=exp.Literal.number(max_results)))
        return root

    limit_expr = limit.args.get("expression")

    if not isinstance(limit_expr, exp.Literal) or not limit_expr.is_number:
        raise ValueError("LIMIT must be a numeric literal.")

    value = int(limit_expr.this)

    if value > max_results:
        limit.set("expression", exp.Literal.number(max_results))

    return root


def validate_metadata_sql(sql: str, max_results: int) -> str | ToolError:
    sql = sql.strip().rstrip(";")

    try:
        expressions = sqlglot.parse(sql, read="duckdb")
    except Exception as e:
        return ToolError(error_type="Invalid SQL", message=str(e))

    if len(expressions) != 1:
        return ToolError(error_type="Invalid SQL", message="Only one SQL statement is allowed.")

    root = expressions[0]

    if not isinstance(root, exp.Select):
        return ToolError(error_type="Invalid SQL", message="Only SELECT queries are allowed.")

    if root.find(*FORBIDDEN_SQL_TYPES):
        return ToolError(error_type="Invalid SQL", message="Only read-only SELECT queries are allowed.")

    for func in root.find_all(exp.Func):
        func_name = func.sql_name().lower()
        if func_name in FORBIDDEN_FUNCTIONS:
            return ToolError(error_type="Invalid SQL", message=f"Disallowed function: {func_name}")

    try:
        root = _qualify_allowed_tables(root)
        root = _enforce_limit(root, max_results)
    except ValueError as e:
        return ToolError(error_type="Invalid SQL", message=str(e))

    return root.sql(dialect="duckdb")


def query_metadata_impl(
    args: dict,
    data_store: DataStore,
    max_results: int,
    include_sql: bool = False,
) -> QueryResult | ToolError:
    try:
        parsed_args = SqlQueryArgs.model_validate(args)
    except ValidationError as e:
        return ToolError(
            error_type="Invalid query_metadata_sql arguments",
            message=str(e),
        )

    sql_or_error = validate_metadata_sql(parsed_args.sql, max_results)
    if isinstance(sql_or_error, ToolError):
        return sql_or_error

    sql = sql_or_error

    try:
        with data_store.connect_duckdb_for_lance() as con:
            rows = con.execute(sql).fetchall()
            columns = [desc[0] for desc in con.description]
    except Exception as e:
        return ToolError(
            error_type="SQL execution error",
            message=str(e),
        )

    return QueryResult(
        columns=columns,
        rows=[dict(zip(columns, row, strict=True)) for row in rows],
        row_count=len(rows),
        sql=sql if include_sql else None,
    )
