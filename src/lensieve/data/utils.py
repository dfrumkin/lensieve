DUCKDB_CATALOG = "lensieve"
DUCKDB_SCHEMA = "main"


def sql_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def sql_ident(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


def duck_table_name(lance_table_name: str) -> str:
    return f"{sql_ident(DUCKDB_CATALOG)}.{sql_ident(DUCKDB_SCHEMA)}.{sql_ident(lance_table_name)}"
