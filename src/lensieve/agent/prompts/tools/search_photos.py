from lensieve.tools.enums import Tool

SEARCH_PHOTOS_TOOL_DESCRIPTION = f"""
Tool: {Tool.SEARCH_PHOTOS.value}

Purpose:
Search local photos by semantic visual query and optional date range.
Use when the user wants to find, view, browse, retrieve, or display actual photos.

Arguments:
{{
  "text_query": string,
  "date_start": string | null,
  "date_end": string | null
}}

Rules:
- text_query is required.
- text_query must describe the visual content to search for.
- text_query should be short and concrete.
- Do not include wrapper phrases such as:
  - "photo of"
  - "image of"
  - "picture of"
  - "show me"
  - "find photos of"
- Use only the visual concepts themselves.
- Examples:
  - "show me photos of dogs" -> "dogs"
  - "images of red cars" -> "red cars"
  - "find beach photos from 2023" -> "beach"
- If the user specifies a year, season, month, relative date, or date range, convert it to date_start/date_end in YYYY-MM-DD format.
- If no date is specified, omit date_start and date_end.
- date_start and date_end are inclusive.
- Do not include unknown fields.

Examples:
- "photos from 2023" -> {{"text_query": "photos", "date_start": "2023-01-01", "date_end": "2023-12-31"}}
- "dogs on the beach" -> {{"text_query": "dogs on the beach"}}
- "photos from last summer" -> infer an appropriate date range.
""".strip()
