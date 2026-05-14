from pydantic import ValidationError

from lensieve.data_store import DataStore
from lensieve.models.model_manager import ModelManager
from lensieve.retrieval.schema import SearchArgs, SearchResult
from lensieve.retrieval.search import search_images


def search_photos_impl(
    args: dict, model_manager: ModelManager, data_store: DataStore, max_results: int
) -> SearchResult | str:
    try:
        args = args | {"max_results": max_results}
        search_args = SearchArgs.model_validate(args)
    except ValidationError as e:
        return f"Invalid search_photos arguments: {e}"
    return search_images(
        args=search_args,
        model_manager=model_manager,
        data_store=data_store,
    )


SEARCH_PHOTOS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_photos",
        "description": "Search local photos by text and date range.",
        "parameters": {
            "type": "object",
            "properties": {
                "text_query": {"type": "string"},
                "date_start": {"type": "string", "description": "YYYY-MM-DD"},
                "date_end": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["text_query"],
        },
    },
}
