import logging
from typing import Any, cast

from llama_cpp import ChatCompletionRequestMessage

from lensieve.agent.parsing import extract_tool_call, serialize_tool_result
from lensieve.agent.prompts import build_photo_agent_system_prompt
from lensieve.data_store import DataStore
from lensieve.models.model_manager import ModelManager
from lensieve.retrieval.schema import SearchResult
from lensieve.tools.search_photos_impl import SEARCH_PHOTOS_TOOL, search_photos_impl

logger = logging.getLogger(__name__)


class PhotoAgent:
    def __init__(
        self,
        model_manager: ModelManager,
        data_store: DataStore,
        northern_hemisphere: bool,
        max_steps: int,
        max_results: int,
    ) -> None:
        self.model_manager = model_manager
        self.data_store = data_store
        self.northern_hemisphere = northern_hemisphere
        self.max_steps = max_steps
        self.max_results = max_results

    def run_once(self, user_query: str) -> str | SearchResult:
        messages: list[ChatCompletionRequestMessage] = [
            {
                "role": "system",
                "content": build_photo_agent_system_prompt(self.northern_hemisphere),
            },
            {
                "role": "user",
                "content": user_query,
            },
        ]

        for step in range(self.max_steps):
            llm = self.model_manager.load_llm()

            logger.debug("LLM step %d", step)

            response = cast(
                dict[str, Any],
                llm.create_chat_completion(
                    messages=messages,
                    tools=[SEARCH_PHOTOS_TOOL],  # type: ignore[arg-type]
                    tool_choice="auto",
                    temperature=0,
                ),
            )

            assistant_msg = response["choices"][0]["message"]
            content = assistant_msg.get("content") or ""

            logger.debug("LLM response: %s", content)

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )

            tool_call = extract_tool_call(content)

            if tool_call is None:
                return content

            logger.info(
                "Tool call: %s(%s)",
                tool_call["name"],
                tool_call["arguments"],
            )

            tool_result = self._run_tool(tool_call)

            logger.debug(
                "Tool result (%s): %s",
                tool_call["name"],
                serialize_tool_result(tool_result)[:500],
            )

            # success → bypass LLM
            if not isinstance(tool_result, str):
                logger.info("Got result from the tool - returning")
                return tool_result

            # error or message → let LLM handle it
            messages.append(
                cast(
                    ChatCompletionRequestMessage,
                    {
                        "role": "tool",
                        "name": tool_call["name"],
                        "content": serialize_tool_result(tool_result),
                    },
                )
            )

        return "Agent stopped: maximum tool steps reached."

    def _run_tool(self, tool_call: dict[str, Any]) -> str | SearchResult:
        name = tool_call["name"]
        args = tool_call["arguments"]

        if not isinstance(args, dict):
            return f"Invalid tool arguments for {name}: expected object, got {type(args).__name__}"

        if name == "search_photos":
            return search_photos_impl(
                args=args, model_manager=self.model_manager, data_store=self.data_store, max_results=self.max_results
            )

        return f"Unknown tool: {name}"
