from dataclasses import dataclass
from pathlib import Path

import duckdb
import lancedb

from lensieve.consts import lancedb_path

DUCKDB_CATALOG = "lensieve"
DUCKDB_SCHEMA = "main"


@dataclass
class Resources:
    root: Path
    lancedb: lancedb.DBConnection
    duckdb: duckdb.DuckDBPyConnection


def connect_lancedb(db_path: Path):
    db_path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(db_path))


def connect_duckdb_for_lance(db_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    # TODO: Do we want to limit memory usage?
    con.execute("INSTALL lance")
    con.execute("LOAD lance")
    con.execute(f"ATTACH '{db_path.as_posix()}' AS {DUCKDB_CATALOG} (TYPE LANCE)")
    return con


def create_resources(root_dir: str | Path) -> Resources:
    root = Path(root_dir).expanduser().resolve()
    db_path = lancedb_path(root)

    return Resources(
        root=root,
        lancedb=connect_lancedb(db_path),
        duckdb=connect_duckdb_for_lance(db_path),
    )


def duck_table_name(lance_table_name: str) -> str:
    return f"{DUCKDB_CATALOG}.{DUCKDB_SCHEMA}.{lance_table_name}"
