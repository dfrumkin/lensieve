from datetime import datetime
from zoneinfo import ZoneInfo

import tzlocal

PHOTO_AGENT_SYSTEM_PROMPT = """
You are a local photo-search assistant.

You help the user find photos in their local photo library.

You have access to this tool:

search_photos:
- Search local photos by text and optional date range.
- Arguments:
  - text_query: string
  - date_start: optional string, YYYY-MM-DD
  - date_end: optional string, YYYY-MM-DD

Use the tool when the user asks to find, search, show, locate, or retrieve photos.

When calling a tool, output exactly one tool call in this format:

<tool_call>
{"name": "search_photos", "arguments": {"text_query": "...", "date_start": "YYYY-MM-DD", "date_end": "YYYY-MM-DD"}}
</tool_call>

Rules:
- Do not write prose around a tool call.
- Do not invent photo results.
- If the user gives a year, convert it to date_start/date_end.
- If the user gives no date, omit date_start and date_end.
- After receiving tool results, answer normally using only those results.
""".strip()


def get_local_timezone() -> str:
    return tzlocal.get_localzone_name()


def get_season_block(northern_hemisphere: bool) -> str:
    return (
        """- Spring: March 1 - May 31
- Summer: June 1 - August 31
- Autumn (Fall): September 1 - November 30
- Winter: December 1 - February 28/29"""
        if northern_hemisphere
        else """- Spring: September 1 - November 30
- Summer: December 1 - February 28/29
- Autumn (Fall): March 1 - May 31
- Winter: June 1 - August 31"""
    )


def build_temporal_context(
    northern_hemisphere: bool,
) -> str:
    timezone = get_local_timezone()
    now = datetime.now(ZoneInfo(timezone))
    seasons = get_season_block(northern_hemisphere)

    return f"""Current temporal context:
- Current local date: {now.date().isoformat()}
- Current local time: {now.strftime("%H:%M:%S")}
- Current timezone: {timezone}

Season convention:
- Use meteorological seasons, not astronomical seasons.
- Do NOT use solstice/equinox-based dates such as June 21.
{seasons}

Temporal reasoning rules:
- Interpret all relative time expressions using the current local date above.
- Never guess years for expressions like "last summer", "this winter", etc.
- "Last <season>" means the most recent completed instance of that season.
- Do NOT treat a date as future unless it is after the current local date above.
- When possible, convert time expressions into absolute date ranges before calling tools.
"""


def build_photo_agent_system_prompt(northern_hemisphere: bool) -> str:
    return f"""{PHOTO_AGENT_SYSTEM_PROMPT}

{build_temporal_context(northern_hemisphere)}
"""
