from enum import StrEnum


class AgentRoute(StrEnum):
    SEARCH_PHOTOS = "search_photos"
    QUERY_METADATA = "query_metadata"
    UNSUPPORTED = "unsupported"
