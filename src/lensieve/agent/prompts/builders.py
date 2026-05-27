from datetime import datetime
from zoneinfo import ZoneInfo

import tzlocal

from lensieve.agent.enums import AgentRoute
from lensieve.agent.prompts.roles.metadata_query import QUERY_METADATA_SYSTEM_PROMPT
from lensieve.agent.prompts.roles.photo_search import SEARCH_PHOTOS_SYSTEM_PROMPT
from lensieve.agent.prompts.roles.repair import REPAIR_TOOL_CALL_SYSTEM_PROMPT
from lensieve.agent.prompts.tools.query_metadata import QUERY_METADATA_TOOL_DESCRIPTION
from lensieve.agent.prompts.tools.search_photos import SEARCH_PHOTOS_TOOL_DESCRIPTION


def _get_local_timezone() -> str:
    return tzlocal.get_localzone_name() or "UTC"


def _get_season_block(northern_hemisphere: bool) -> str:
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


def _build_temporal_context(
    northern_hemisphere: bool,
) -> str:
    timezone = _get_local_timezone()
    now = datetime.now(ZoneInfo(timezone))
    seasons = _get_season_block(northern_hemisphere)

    return f"""Current temporal context:
- Current local date: {now.date().isoformat()}
- Current local time: {now.strftime("%H:%M:%S")}
- Current timezone: {timezone}

Season convention:
- Use meteorological seasons, not astronomical seasons.
- Do NOT use solstice/equinox-based dates such as June 21.
{seasons}

Temporal reasoning rules:
- Interpret relative time expressions using the current local date above.
- "this <period>" = current calendar <period>.
- "last <period>" = most recent completed <period>.
- "last N years/months/days" = rolling interval from current local date minus N years/months/days through the current local date.
- Explicitly exclude future dates when asked about past time periods.
- When possible, convert relative time expressions into absolute date ranges before calling tools.
"""


def _build_tools_block(agent_route: AgentRoute) -> str:
    match agent_route:
        case AgentRoute.SEARCH_PHOTOS:
            tool_descriptions = [SEARCH_PHOTOS_TOOL_DESCRIPTION]
        case AgentRoute.QUERY_METADATA:
            tool_descriptions = [QUERY_METADATA_TOOL_DESCRIPTION]
        case _:
            raise ValueError(f"Unsupported agent route: {agent_route}")

    return "\n\n".join(["Available tools:", *tool_descriptions]).strip()


def _get_route_worker_prompt(agent_route: AgentRoute) -> str:
    match agent_route:
        case AgentRoute.SEARCH_PHOTOS:
            return SEARCH_PHOTOS_SYSTEM_PROMPT
        case AgentRoute.QUERY_METADATA:
            return QUERY_METADATA_SYSTEM_PROMPT
        case _:
            raise ValueError(f"Unsupported agent route: {agent_route}")


def _join_prompt_sections(*sections: str) -> str:
    return "\n\n".join(s.strip() for s in sections if s and s.strip())


def _build_prompt(agent_route: AgentRoute, role_prompt: str, northern_hemisphere: bool) -> str:
    tools_block = _build_tools_block(agent_route)
    temporal_context = _build_temporal_context(northern_hemisphere)

    return _join_prompt_sections(
        role_prompt,
        temporal_context,
        tools_block,
    )


def build_tool_worker_system_prompt(
    agent_route: AgentRoute,
    northern_hemisphere: bool,
) -> str:
    role_prompt = _get_route_worker_prompt(agent_route)
    return _build_prompt(agent_route, role_prompt, northern_hemisphere)


def build_tool_repair_system_prompt(agent_route: AgentRoute, northern_hemisphere: bool) -> str:
    return _build_prompt(agent_route, REPAIR_TOOL_CALL_SYSTEM_PROMPT, northern_hemisphere)
