from lensieve.data.orientation import (
    ASPECT_RATIO_BUCKETS,
    ASPECT_RATIO_EPSILON,
    SWAPS_WIDTH_HEIGHT,
    AspectRatio,
    OrientationKind,
)
from lensieve.data.utils import duck_table_name, sql_ident, sql_quote
from lensieve.names import ImageField as IF
from lensieve.names import ImageViewField as IVF
from lensieve.names import TableName as TN

_SWAPS_WIDTH_HEIGHT_SQL = ", ".join(f"{sql_quote(value)}" for value in SWAPS_WIDTH_HEIGHT)
_ASPECT_VALUE = "aspect_value"


def _orientation_kind_sql() -> str:
    width: str = sql_ident(IVF.DISPLAY_WIDTH)
    height: str = sql_ident(IVF.DISPLAY_HEIGHT)
    return f"""
        CASE
        WHEN {width} > {height} THEN {sql_quote(OrientationKind.LANDSCAPE.value)}
        WHEN {width} < {height} THEN {sql_quote(OrientationKind.PORTRAIT.value)}
        ELSE {sql_quote(OrientationKind.SQUARE.value)}
        END
    """.strip()


def _display_dim_sql(*, width_dim: bool) -> str:
    width = sql_ident(IF.WIDTH)
    height = sql_ident(IF.HEIGHT)
    normal_dim, swapped_dim = (width, height) if width_dim else (height, width)
    orientation = sql_ident(IF.ORIENTATION)

    return f"""
    CASE
      WHEN {orientation} IN ({_SWAPS_WIDTH_HEIGHT_SQL}) THEN {swapped_dim}
      ELSE {normal_dim}
    END
    """.strip()


def _display_width_sql() -> str:
    return _display_dim_sql(width_dim=True)


def _display_height_sql() -> str:
    return _display_dim_sql(width_dim=False)


def _aspect_value_sql() -> str:
    width = sql_ident(IVF.DISPLAY_WIDTH)
    height = sql_ident(IVF.DISPLAY_HEIGHT)
    return f"{width}::DOUBLE / NULLIF({height}::DOUBLE, 0)"


def _aspect_ratio_sql() -> str:
    aspect = sql_ident(_ASPECT_VALUE)
    cases = [
        f"""
        WHEN {aspect} IS NULL THEN NULL
        WHEN ABS(({aspect}) - 1.0) < {ASPECT_RATIO_EPSILON}
        THEN {sql_quote(AspectRatio.RATIO_1_1.value)}
        """.strip()
    ]

    for target, landscape_label, portrait_label in ASPECT_RATIO_BUCKETS:
        cases.append(
            f"""
            WHEN ABS(({aspect}) - {target}) < {ASPECT_RATIO_EPSILON}
            THEN {sql_quote(landscape_label)}
            """.strip()
        )
        cases.append(
            f"""
            WHEN ABS(({aspect}) - {1 / target}) < {ASPECT_RATIO_EPSILON}
            THEN {sql_quote(portrait_label)}
            """.strip()
        )

    cases.append(
        f"""
        WHEN ({aspect}) > 1.0 THEN {sql_quote(AspectRatio.LANDSCAPE_OTHER.value)}
        """.strip()
    )
    cases.append(
        f"""
        WHEN ({aspect}) < 1.0 THEN {sql_quote(AspectRatio.PORTRAIT_OTHER.value)}
        """.strip()
    )

    return f"""
    CASE
      {" ".join(cases)}
      ELSE {sql_quote(AspectRatio.RATIO_1_1.value)}
    END
    """.strip()


IMAGES_VIEW = f"""
CREATE OR REPLACE VIEW {duck_table_name(TN.IMAGES_VIEW)} AS
WITH base AS (
    SELECT
        {sql_ident(IF.PATH)},
        {sql_ident(IF.SHA256)},
        {sql_ident(IF.DATE_TAKEN)},
        {sql_ident(IF.CAMERA_MAKE)},
        {sql_ident(IF.CAMERA_MODEL)},
        {sql_ident(IF.IMAGE_FORMAT)},
        {sql_ident(IF.EXIF_ERROR)},
        {sql_ident(IF.STRUCTURE_ERROR)},
        {sql_ident(IF.WIDTH)},
        {sql_ident(IF.HEIGHT)},
        {sql_ident(IF.ORIENTATION)},
        {_display_width_sql()} AS {sql_ident(IVF.DISPLAY_WIDTH)},
        {_display_height_sql()} AS {sql_ident(IVF.DISPLAY_HEIGHT)}
    FROM {duck_table_name(TN.IMAGES)}
),
with_aspect AS (
    SELECT
        *,
        {_aspect_value_sql()} AS {sql_ident(_ASPECT_VALUE)}
    FROM base
)
SELECT
    * EXCLUDE ({sql_ident(_ASPECT_VALUE)}),
    {_orientation_kind_sql()} AS {sql_ident(IVF.ORIENTATION_KIND)},
    {_aspect_ratio_sql()} AS {sql_ident(IVF.ASPECT_RATIO)}
FROM with_aspect
""".strip()
