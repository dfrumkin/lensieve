from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class SearchArgs(BaseModel):
    text_query: str | None = None
    image_query_path: Path | None = None
    image_query_sha256: str | None = None
    date_start: date | None = None
    date_end: date | None = None
    max_results: int = Field(default=30, gt=0)

    @model_validator(mode="after")
    def exactly_one_query(self):
        provided = [
            self.text_query is not None,
            self.image_query_path is not None,
            self.image_query_sha256 is not None,
        ]

        if sum(provided) != 1:
            raise ValueError("Exactly one of text_query, image_query_path, or image_query_sha256 must be provided")

        return self

    @model_validator(mode="after")
    def valid_date_range(self):
        if self.date_start and self.date_end and self.date_end < self.date_start:
            raise ValueError("date_end must be >= date_start")
        return self


class ImageHit(BaseModel):
    path: str
    sha256: str
    score: float


class SearchResult(BaseModel):
    hits: list[ImageHit]
    similarity_matrix: list[list[float]]
