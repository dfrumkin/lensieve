from lensieve.agent.prompts.tools.common import ASPECT_RATIO_VALUES, IMAGE_FORMAT_VALUES, ORIENTATION_KIND_VALUES
from lensieve.data.orientation import AspectRatio, OrientationKind
from lensieve.names import ImageViewField as IVF
from lensieve.names import TableName as TN
from lensieve.tools.enums import Tool

QUERY_METADATA_SQL_TOOL_DESCRIPTION = f"""
Tool: {Tool.QUERY_METADATA_SQL.value}

Purpose:
Answer factual and statistical questions about the local image collection metadata.

Arguments:
{{
  "sql": "<SQL query>"
}}

Rules:
- Only SELECT queries are allowed.
- Query only from {TN.IMAGES_VIEW}.
- Do not use INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, COPY, ATTACH, DETACH, or other modifying statements.
- Do not use file-reading functions such as read_csv, read_parquet, read_json, glob, or similar functions.
- Prefer explicit aliases for computed columns.
- Add LIMIT when the result may be large.
- Use YYYY-MM-DD format for date filters.
- For time-based queries, usually exclude NULL {IVF.DATE_TAKEN} values.
- When possible, prefer using {IVF.ORIENTATION_KIND} and {IVF.ASPECT_RATIO} for orientation and shape queries.
- Never assume that {AspectRatio.LANDSCAPE_OTHER.value} or {AspectRatio.PORTRAIT_OTHER.value} represent panoramas or any loosely defined category.
- Default panorama condition: ({IVF.DISPLAY_WIDTH} >= 2.5 * {IVF.DISPLAY_HEIGHT} OR {IVF.DISPLAY_HEIGHT} >= 2.5 * {IVF.DISPLAY_WIDTH})
- When using {IVF.DISPLAY_WIDTH}, {IVF.DISPLAY_HEIGHT}, {IVF.ORIENTATION_KIND}, or {IVF.ASPECT_RATIO}, remember they may be NULL if dimensions are unavailable.
- When using GROUP BY, always include ORDER BY unless the user explicitly requests a different ordering or no ordering.
- Default ordering for grouped aggregate queries:
  - Primary sort: descending by the main aggregate value.
  - Secondary sort: ascending by the grouping columns for deterministic output.

Schema:

{IVF.PATH}: STRING NOT NULL
- Relative image path or filename.

{IVF.SHA256}: STRING NULL
- SHA-256 hash of the image file.
- NULL means hash generation failed or the file could not be hashed.
- Duplicate files share the same {IVF.SHA256} value.

{IVF.DATE_TAKEN}: TIMESTAMP NULL
- Timestamp when the image was taken, extracted from EXIF metadata.
- NULL means the image has no usable capture timestamp.

{IVF.CAMERA_MAKE}: STRING NULL
- Camera manufacturer from EXIF metadata.
- NULL means unavailable or missing in EXIF metadata.

{IVF.CAMERA_MODEL}: STRING NULL
- Camera or device model from EXIF metadata.
- NULL means unavailable or missing in EXIF metadata.

{IVF.IMAGE_FORMAT}: STRING
- Image file format.
- Possible values:
{IMAGE_FORMAT_VALUES}
- NULL means the image format could not be determined because of a structure/loading error.

{IVF.EXIF_ERROR}: BOOLEAN NOT NULL
- TRUE means EXIF parsing failed or produced an error.
- FALSE means no EXIF parsing error was reported.

{IVF.STRUCTURE_ERROR}: BOOLEAN NOT NULL
- TRUE means image loading or structural validation failed.
- FALSE means the image structure was parsed successfully.

{IVF.DISPLAY_WIDTH}: INTEGER NULL
- Displayed image width in pixels after applying EXIF orientation correction.
- NULL means the display width could not be determined.

{IVF.DISPLAY_HEIGHT}: INTEGER NULL
- Displayed image height in pixels after applying EXIF orientation correction.
- NULL means the display height could not be determined.

{IVF.ORIENTATION_KIND}: STRING NULL
- Simplified orientation category computed from display dimensions.
- Possible values:
{ORIENTATION_KIND_VALUES}
- NULL means display dimensions are unavailable.

{IVF.ASPECT_RATIO}: STRING NULL
- Simplified aspect-ratio category computed from display dimensions.
- Only the rectangular image frame proportions, not shapes inside the image or non-rectangular masks.
- Possible values:
{ASPECT_RATIO_VALUES}
- NULL means display dimensions are unavailable.

Example queries:

Count all photos:
SELECT COUNT(*) AS num_images
FROM {TN.IMAGES_VIEW}

Count photos by year:
SELECT
    YEAR({IVF.DATE_TAKEN}) AS year,
    COUNT(*) AS num_images
FROM {TN.IMAGES_VIEW}
WHERE {IVF.DATE_TAKEN} IS NOT NULL
GROUP BY year
ORDER BY year

Count photos by image format during Spring 2024:
SELECT
    {IVF.IMAGE_FORMAT},
    COUNT(*) AS num_images
FROM {TN.IMAGES_VIEW}
WHERE {IVF.DATE_TAKEN} BETWEEN '2024-03-01' AND '2024-05-31'
GROUP BY {IVF.IMAGE_FORMAT}
ORDER BY num_images DESC

Count portrait images:
SELECT
    COUNT(*) AS num_images
FROM {TN.IMAGES_VIEW}
WHERE {IVF.ORIENTATION_KIND} = '{OrientationKind.PORTRAIT.value}'

Count photos by aspect ratio:
SELECT
    {IVF.ASPECT_RATIO},
    COUNT(*) AS num_images
FROM {TN.IMAGES_VIEW}
WHERE {IVF.ASPECT_RATIO} IS NOT NULL
GROUP BY {IVF.ASPECT_RATIO}
ORDER BY num_images DESC

Find the largest images:
SELECT
    {IVF.PATH},
    {IVF.DISPLAY_WIDTH},
    {IVF.DISPLAY_HEIGHT}
FROM {TN.IMAGES_VIEW}
WHERE {IVF.DISPLAY_WIDTH} IS NOT NULL AND {IVF.DISPLAY_HEIGHT} IS NOT NULL
ORDER BY {IVF.DISPLAY_WIDTH} * {IVF.DISPLAY_HEIGHT} DESC
LIMIT 20
""".strip()
