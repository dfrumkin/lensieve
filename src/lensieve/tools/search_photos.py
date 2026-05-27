from pydantic import ValidationError

from lensieve.data.data_store import DataStore
from lensieve.models.model_manager import ModelManager
from lensieve.retrieval.schema import SearchArgs, SearchResult
from lensieve.retrieval.search import search_images
from lensieve.tools.errors import ToolError


def search_photos_impl(
    args: dict, model_manager: ModelManager, data_store: DataStore, max_results: int
) -> SearchResult | ToolError:
    try:
        args = args | {"max_results": max_results}
        search_args = SearchArgs.model_validate(args)
    except ValidationError as e:
        return ToolError(error_type="Invalid search_photos arguments", message=str(e))
    return search_images(
        args=search_args,
        model_manager=model_manager,
        data_store=data_store,
    )
