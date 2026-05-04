from itertools import batched
from logging import Logger

import lancedb
import pyarrow as pa


def open_or_create_table(
    db: lancedb.DBConnection,
    table_name: str,
    schema: pa.Schema,
    logger: Logger,
) -> lancedb.table.Table:
    existing_tables = set(db.table_names())

    if table_name in existing_tables:
        logger.info("Existing %s table found; opening", table_name)
        table = db.open_table(table_name)

        if not table.schema.equals(schema, check_metadata=False):
            raise ValueError(
                f"Existing {table_name!r} table has incompatible schema.\nExpected:\n{schema}\n\nFound:\n{table.schema}"
            )

        return table

    logger.info("No existing %s table found; creating new one", table_name)
    return db.create_table(table_name, schema=schema)


def sql_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def sql_ident(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


def delete_rows(table: lancedb.Table, col: str, vals: list[str], logger: Logger, batch_size: int = 1_000) -> None:
    if not vals:
        return
    logger.info("Deleting %s removed/changed rows from %s table", len(vals), table.name)
    col_expr = sql_ident(col)
    for batch in batched(vals, batch_size, strict=False):
        values = ", ".join(sql_quote(p) for p in batch)
        table.delete(f"{col_expr} IN ({values})")


def insert_rows(table: lancedb.Table, rows: list[dict], logger: Logger, batch_size: int = 10_000) -> None:
    if not rows:
        return
    logger.info("Inserting %s new/updated rows into %s table", len(rows), table.name)
    for batch in batched(rows, batch_size, strict=False):
        table.add(batch)
