from pathlib import Path

import duckdb
import lancedb

from lensieve.data.images_view import IMAGES_VIEW
from lensieve.data.utils import DUCKDB_CATALOG, sql_quote
from lensieve.names import lancedb_path


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
        con.execute(IMAGES_VIEW)
        return con
