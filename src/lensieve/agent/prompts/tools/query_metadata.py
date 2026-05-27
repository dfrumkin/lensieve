from lensieve.agent.prompts.tools.common import ASPECT_RATIO_VALUES, IMAGE_FORMAT_VALUES, ORIENTATION_KIND_VALUES
from lensieve.query.spec import LIST_FILTER_OPS, SIMPLE_FILTER_OPS, DatePart, FilterOp
from lensieve.query.spec import AggregateOp as AggOp
from lensieve.query.spec import OrderDirection as OrderDir
from lensieve.query.spec import QueryField as QF
from lensieve.tools.enums import Tool

_AGG_OP_VALUES = "\n".join(f"- {op.value}" for op in AggOp)
_DATE_PART_VALUES = "\n".join(f"- {part.value}" for part in DatePart)
_SIMPLE_FILTER_OPS_VALUES = "\n".join(f"- {op.value}" for op in SIMPLE_FILTER_OPS)
_LIST_FILTER_OPS_VALUES = "\n".join(f"- {op.value}" for op in LIST_FILTER_OPS)
_ORDER_DIR_VALUES = "\n".join(f"- {dir.value}" for dir in OrderDir)

QUERY_METADATA_TOOL_DESCRIPTION = f"""
Tool: {Tool.QUERY_METADATA.value}

Purpose:
Answer factual and statistical questions about the local image collection metadata.

Arguments:
{{
  "select": list[SelectExpr],
  "filters": list[Filter] = [],
  "group_by": list[str] = [],
  "order_by": list[OrderBy] = [],
  "limit": int = 100
}}

Rules:
- select must contain at least one item
- every select expression must have an alias
- aliases must match:
  [A-Za-z_][A-Za-z0-9_]*
- filters reference QueryField values only, never select aliases or derived fields
- group_by and order_by reference select aliases, not raw column names
- select aliases may only be referenced from group_by and order_by
- limit must be between 1 and 1000

SelectExpr:
{{
  "expr": Expr,
  "alias": "some_name"
}}

Expr types:

1. ColumnExpr

Format:
{{
  "column": QueryField
}}

Example:
{{
  "expr": {{
    "column": "{QF.IMAGE_FORMAT.value}"
  }},
  "alias": "image_format"
}}

2. AggregateExpr

Format:
{{
  "op": AggregateOp,
  "column": QueryField
}}

For row count:
{{
  "op": "{AggOp.COUNT.value}"
}}

Examples:
{{
  "expr": {{
    "op": "{AggOp.COUNT.value}"
  }},
  "alias": "num_images"
}}

{{
  "expr": {{
    "op": "{AggOp.COUNT_DISTINCT.value}",
    "column": "{QF.SHA256.value}"
  }},
  "alias": "num_unique"
}}

{{
  "expr": {{
    "op": "{AggOp.COUNT_DUPLICATES.value}",
    "column": "{QF.SHA256.value}"
  }},
  "alias": "num_duplicates"
}}

AggregateOp values:
{_AGG_OP_VALUES}

Rules:
- count does not require column
- all other aggregate ops require column
- {AggOp.COUNT_DUPLICATES.value} means:
  {AggOp.COUNT.value}(column) - {AggOp.COUNT_DISTINCT.value}(column)

3. DatePartExpr

Format:
{{
  "part": DatePartExpr,
  "column": "{QF.DATE_TAKEN.value}"
}}

Example:
{{
  "expr": {{
    "part": "{DatePart.YEAR.value}",
    "column": "{QF.DATE_TAKEN.value}"
  }},
  "alias": "year"
}}

Supported date parts:
{_DATE_PART_VALUES}

Use DatePartExpr for grouping by calendar time. Never filter on DatePartExpr aliases such as year or month; filter date_taken using explicit date ranges instead.

Filter:
{{
  "column": QueryField,
  "op": FilterOp,
  "value": ...
}}

Examples:

{{
  "column": "{QF.IMAGE_FORMAT.value}",
  "op": "{FilterOp.EQ.value}",
  "value": "JPEG"
}}

{{
  "column": "{QF.DATE_TAKEN.value}",
  "op": "{FilterOp.BETWEEN.value}",
  "value": ["2023-01-01", "2023-12-31"]
}}

{{
  "column": "{QF.DATE_TAKEN.value}",
  "op": "{FilterOp.NE.value}",
  "value": null
}}

FilterOp values:

Comparison:
{_SIMPLE_FILTER_OPS_VALUES}

List:
{_LIST_FILTER_OPS_VALUES}

Range:
- {FilterOp.BETWEEN.value}

Text:
- {FilterOp.CONTAINS.value}

Rules:
- column must be a QueryField value
- column must not be a select alias such as year, month, day, date, or num_images
- {FilterOp.IN.value}/{FilterOp.NOT_IN.value} require non-empty list
- {FilterOp.BETWEEN.value} requires exactly two values

OrderBy:
{{
  "expr": "alias_name",
  "direction": OrderBy
}}

OrderBy values:
{_ORDER_DIR_VALUES}

QueryField values:
- {QF.PATH.value}: image file path
- {QF.SHA256.value}: content hash for exact duplicate detection
- {QF.DATE_TAKEN.value}: image timestamp, may be null
- {QF.CAMERA_MAKE.value}: camera manufacturer
- {QF.CAMERA_MODEL.value}: camera model
- {QF.IMAGE_FORMAT.value}:
{IMAGE_FORMAT_VALUES}
- {QF.EXIF_ERROR.value}: boolean, true if EXIF parsing failed
- {QF.STRUCTURE_ERROR.value}: boolean, true if image loading/parsing failed
- {QF.DISPLAY_WIDTH.value}: oriented display width
- {QF.DISPLAY_HEIGHT.value}: oriented display height
- {QF.ORIENTATION_KIND}:
{ORIENTATION_KIND_VALUES}
- {QF.ASPECT_RATIO.value}:
{ASPECT_RATIO_VALUES}

Notes:
- Images with the same {QF.SHA256.value} are exact duplicates
- For date statistics, usually filter:
  {{"column": "{QF.DATE_TAKEN.value}", "op": "{FilterOp.NE.value}", "value": null}}
- {QF.ORIENTATION_KIND.value} is coarse orientation
- {QF.ASPECT_RATIO.value} is more detailed
- When using group_by, usually include a count aggregate.
- For ranking groups, order_by should usually use the aggregate alias, not the group key.

Common query patterns:

Total image count:
{{
  "select": [
    {{
      "expr": {{
        "op": "{AggOp.COUNT.value}"
      }},
      "alias": "num_images"
    }}
  ]
}}

Duplicate image count:
{{
  "select": [
    {{
      "expr": {{
        "op": "{AggOp.COUNT_DUPLICATES.value}",
        "column": "{QF.SHA256.value}"
      }},
      "alias": "num_duplicates"
    }}
  ]
}}

Count by year:
{{
  "select": [
    {{
      "expr": {{
        "part": "{DatePart.YEAR.value}",
        "column": "{QF.DATE_TAKEN.value}"
      }},
      "alias": "year"
    }},
    {{
      "expr": {{
        "op": "{AggOp.COUNT.value}"
      }},
      "alias": "num_images"
    }}
  ],
  "filters": [
    {{
      "column": "{QF.DATE_TAKEN.value}",
      "op": "{FilterOp.NE.value}",
      "value": null
    }}
  ],
  "group_by": ["year"],
  "order_by": [
    {{
      "expr": "year",
      "direction": "{OrderDir.ASC.value}"
    }}
  ]
}}

Count by image format:
{{
  "select": [
    {{
      "expr": {{
        "column": "{QF.IMAGE_FORMAT.value}"
      }},
      "alias": "image_format"
    }},
    {{
      "expr": {{
        "op": "{AggOp.COUNT.value}"
      }},
      "alias": "num_images"
    }}
  ],
  "group_by": ["image_format"],
  "order_by": [
    {{
      "expr": "num_images",
      "direction": "{OrderDir.DESC.value}"
    }}
  ]
}}

Output:
{{
  "columns": list[str],
  "rows": list[dict[str, Any]],
  "row_count": int
}}

Meaning:
- columns: output column names
- rows: result rows
- row_count: number of returned rows
""".strip()
