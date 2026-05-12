import logging
from itertools import batched

import lancedb
import pyarrow as pa

from lensieve.data_store import sql_ident, sql_quote

logger = logging.getLogger(__name__)


def open_or_create_table(
    db: lancedb.DBConnection,
    table_name: str,
    schema: pa.Schema,
    from_scratch: bool,
) -> lancedb.table.Table:
    existing_tables = set(db.table_names())

    if table_name in existing_tables:
        logger.info("Existing %s table found", table_name)
        if from_scratch:
            logger.info("Dropping %s table", table_name)
            db.drop_table(table_name)
        else:
            logger.info("Opening %s table", table_name)
            table = db.open_table(table_name)

            if not table.schema.equals(schema, check_metadata=False):
                raise ValueError(
                    f"Existing {table_name!r} table has incompatible schema.\n"
                    f"Expected:\n{schema}\n\nFound:\n{table.schema}"
                )

            return table

    logger.info("Creating %s table", table_name)
    return db.create_table(table_name, schema=schema)


def delete_rows(table: lancedb.Table, col: str, vals: list[str], batch_size: int = 1_000) -> None:
    if not vals:
        return
    logger.info("Deleting %s removed/changed rows from %s table", len(vals), table.name)
    col_expr = sql_ident(col)
    for batch in batched(vals, batch_size, strict=False):
        values = ", ".join(sql_quote(p) for p in batch)
        table.delete(f"{col_expr} IN ({values})")


def insert_rows(table: lancedb.Table, rows: list[dict], batch_size: int = 10_000) -> None:
    if not rows:
        return
    logger.info("Inserting %s new/updated rows into %s table", len(rows), table.name)
    for batch in batched(rows, batch_size, strict=False):
        table.add(pa.Table.from_pylist(batch))
