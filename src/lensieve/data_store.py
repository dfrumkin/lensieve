from pathlib import Path

import duckdb
import lancedb

from lensieve.names import lancedb_path

DUCKDB_CATALOG = "lensieve"
DUCKDB_SCHEMA = "main"


def sql_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def sql_ident(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


def duck_table_name(lance_table_name: str) -> str:
    return f"{sql_ident(DUCKDB_CATALOG)}.{sql_ident(DUCKDB_SCHEMA)}.{sql_ident(lance_table_name)}"


class DataStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root = Path(root_dir).expanduser().resolve()

        self.db_path = lancedb_path(self.root)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.lancedb = lancedb.connect(self.db_path)

        with duckdb.connect(database=":memory:") as con:
            con.execute("INSTALL lance")

    def connect_duckdb_for_lance(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(database=":memory:")
        con.execute("LOAD lance")
        con.execute(f"ATTACH {sql_quote(self.db_path.as_posix())} AS {DUCKDB_CATALOG} (TYPE LANCE)")
        return con
