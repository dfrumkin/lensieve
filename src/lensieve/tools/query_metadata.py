from pydantic import ValidationError

from lensieve.data.data_store import DataStore
from lensieve.query.engine import run_query
from lensieve.query.spec import QueryResult, QuerySpec
from lensieve.tools.errors import ToolError


def query_metadata_impl(
    args: dict,
    data_store: DataStore,
    include_sql: bool = False,
) -> QueryResult | ToolError:
    try:
        spec = QuerySpec.model_validate(args)
    except ValidationError as e:
        return ToolError(error_type="Invalid query_metadata arguments", message=str(e))

    return run_query(
        data_store=data_store,
        spec=spec,
        include_sql=include_sql,
    )
