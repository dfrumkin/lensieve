import json
import logging
import time
from typing import Any, cast

from llama_cpp import ChatCompletionRequestMessage

from lensieve.agent.enums import AgentRoute
from lensieve.agent.parsing import ToolCall, extract_tool_call, serialize_tool_result
from lensieve.agent.prompts.builders import build_tool_repair_system_prompt, build_tool_worker_system_prompt
from lensieve.agent.prompts.roles.metadata_answer import QUERY_METADATA_ANSWER_SYSTEM_PROMPT
from lensieve.agent.prompts.roles.router import ROUTER_SYSTEM_PROMPT
from lensieve.data.data_store import DataStore
from lensieve.models.model_manager import ModelManager, ModelRole
from lensieve.retrieval.schema import SearchResult
from lensieve.tools.enums import Tool
from lensieve.tools.errors import ToolError
from lensieve.tools.query_metadata_sql import QueryResult, query_metadata_sql_impl
from lensieve.tools.search_photos import search_photos_impl

logger = logging.getLogger(__name__)

UNSUPPORTED_MESSAGE = "I can help search your photo library or answer questions about photo metadata and statistics."
ToolResult = str | SearchResult | QueryResult
SuccessfulToolRun = tuple[ToolCall, ToolResult]
ToolExecutionResult = SuccessfulToolRun | ToolError


def _build_messages(system_prompt: str, user_query: str) -> list[ChatCompletionRequestMessage]:
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]


class PhotoAgent:
    def __init__(
        self,
        model_manager: ModelManager,
        data_store: DataStore,
        northern_hemisphere: bool,
        max_image_results: int,
        max_metadata_results: int,
    ) -> None:
        self.model_manager = model_manager
        self.data_store = data_store
        self.northern_hemisphere = northern_hemisphere
        self.max_image_results = max_image_results
        self.max_metadata_results = max_metadata_results

    def _call_llm(
        self,
        llm_role: ModelRole,
        messages: list[ChatCompletionRequestMessage],
    ) -> str:
        t0 = time.perf_counter()
        llm = self.model_manager.load_llm(llm_role)
        llm_info = self.model_manager.get_llm_info(llm_role)

        t1 = time.perf_counter()

        response = cast(
            dict[str, Any],
            llm.create_chat_completion(
                messages=messages,
                temperature=0,
                max_tokens=llm_info.max_tokens,
            ),
        )

        t2 = time.perf_counter()

        usage = response.get("usage", {})
        logger.info(
            "LLM timing (%s): load=%.2fs call=%.2fs usage=%s",
            llm_role,
            t1 - t0,
            t2 - t1,
            usage,
        )

        resp = response["choices"][0]["message"].get("content", "")
        logger.info("LLM response: %s", resp)
        return resp

    def _route(self, user_query: str) -> AgentRoute:
        messages = _build_messages(
            system_prompt=ROUTER_SYSTEM_PROMPT,
            user_query=user_query,
        )
        content = self._call_llm(ModelRole.ROUTER, messages)

        try:
            data = json.loads(content)
            route = data["route"]
            return AgentRoute(route)

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.error("Router failed. content=%r error=%s", content, e)
            return AgentRoute.UNSUPPORTED

    def _generate_tool_call(
        self,
        *,
        user_query: str,
        agent_route: AgentRoute,
        llm_role: ModelRole,
    ) -> ToolCall | None:
        system_prompt = build_tool_worker_system_prompt(
            agent_route,
            self.northern_hemisphere,
        )
        messages = _build_messages(
            system_prompt=system_prompt,
            user_query=user_query,
        )

        content = self._call_llm(llm_role, messages)
        tool_call = extract_tool_call(content)

        if tool_call is None:
            logger.error("Worker did not produce a tool call. content=%r", content)
            return None

        logger.info("Generated tool call: %s", tool_call.to_dict())
        return tool_call

    def _run_tool(self, tool_call: ToolCall) -> ToolResult | ToolError:
        name = tool_call.tool
        args = tool_call.arguments

        logger.info("Tool call: %s(%s)", name, args)

        match name:
            case Tool.SEARCH_PHOTOS:
                result = search_photos_impl(
                    args=args,
                    model_manager=self.model_manager,
                    data_store=self.data_store,
                    max_results=self.max_image_results,
                )
            case Tool.QUERY_METADATA_SQL:
                result = query_metadata_sql_impl(
                    args=args, data_store=self.data_store, max_results=self.max_metadata_results
                )
            case _:
                result = ToolError(error_type="Cannot run the tool", message=f"Unknown tool: {name}")

        logger.debug(
            "Tool result (%s): %s",
            name,
            serialize_tool_result(result)[:500],
        )

        return result

    def _repair_tool_call(
        self,
        *,
        user_query: str,
        llm_role: ModelRole,
        agent_route: AgentRoute,
        bad_tool_call: ToolCall,
        error: ToolError,
    ) -> ToolCall | None:
        system_prompt = build_tool_repair_system_prompt(agent_route, self.northern_hemisphere)
        bad_tool_call_json = json.dumps(bad_tool_call.to_dict(), ensure_ascii=False)
        repair_user_query = (
            "Original user request:\n"
            f"{user_query}\n\n"
            "Invalid tool call:\n"
            f"{bad_tool_call_json}\n\n"
            "Validation/tool error:\n"
            f"{error.message}\n\n"
            "Return a corrected tool call."
        )
        messages = _build_messages(system_prompt=system_prompt, user_query=repair_user_query)
        content = self._call_llm(llm_role, messages)
        return extract_tool_call(content)

    def _run_tool_with_optional_repair(
        self,
        *,
        user_query: str,
        agent_route: AgentRoute,
        tool_call: ToolCall,
        repair_llm_role: ModelRole,
    ) -> ToolExecutionResult:
        result = self._run_tool(tool_call)

        if not isinstance(result, ToolError):
            return tool_call, result

        logger.error("Tool error: %s", result.message)
        logger.info("Attempting to repair tool call")

        repaired_tool_call = self._repair_tool_call(
            user_query=user_query,
            llm_role=repair_llm_role,
            agent_route=agent_route,
            bad_tool_call=tool_call,
            error=result,
        )

        if repaired_tool_call is None:
            logger.error("Failed to repair tool call")
            return result

        logger.info("Repaired tool call: %s", repaired_tool_call)

        repaired_result = self._run_tool(repaired_tool_call)

        if isinstance(repaired_result, ToolError):
            logger.error("Repaired tool call also failed: %s", repaired_result.message)
            return repaired_result

        return repaired_tool_call, repaired_result

    def _run_search_photos_pipeline(self, user_query: str) -> SearchResult | str:
        tool_call = self._generate_tool_call(
            user_query=user_query,
            agent_route=AgentRoute.SEARCH_PHOTOS,
            llm_role=ModelRole.IMAGE_RETRIEVAL,
        )

        if tool_call is None:
            logger.warning("Unsupported photo search query: %s", user_query)
            return "Sorry, I could not convert the request into a photo search.  Try asking in a different way."

        run_result = self._run_tool_with_optional_repair(
            user_query=user_query,
            agent_route=AgentRoute.SEARCH_PHOTOS,
            tool_call=tool_call,
            repair_llm_role=ModelRole.IMAGE_RETRIEVAL_REPAIR,
        )

        if isinstance(run_result, ToolError):
            logger.error("Photo search failed: %s\n%s: %s", user_query, run_result.error_type, run_result.message)
            return "Sorry, something went wrong while searching photos."

        _, result = run_result
        if isinstance(result, SearchResult):
            logger.info("Photo search successful: %s results", len(result.hits))
            return result

        logger.error("Photo search failed: %s\n%s", user_query, result)
        return "Sorry, something went wrong while searching photos."

    def _interpret_metadata_result(
        self,
        *,
        user_query: str,
        tool_call: ToolCall,
        result: QueryResult | str,
    ) -> str:
        logger.info("Interpreting metadata query result")
        tool_result_text = serialize_tool_result(result)

        interpretation_user_query = (
            "Original user request:\n"
            f"{user_query}\n\n"
            "Tool call that was executed:\n"
            f"{json.dumps(tool_call.to_dict(), ensure_ascii=False)}\n\n"
            "Tool result:\n"
            f"{tool_result_text}\n\n"
            "Answer the original user request using only the tool result."
        )

        messages = _build_messages(
            system_prompt=QUERY_METADATA_ANSWER_SYSTEM_PROMPT,
            user_query=interpretation_user_query,
        )

        return self._call_llm(ModelRole.METADATA_INTERPRETER, messages)

    def _run_metadata_query_pipeline(self, user_query: str) -> str:
        tool_call = self._generate_tool_call(
            user_query=user_query,
            agent_route=AgentRoute.QUERY_METADATA,
            llm_role=ModelRole.METADATA_QUERY,
        )

        if tool_call is None:
            logger.warning("Unsupported metadata query: %s", user_query)
            return "Sorry, I could not convert the request into a metadata query.  Try asking in a different way."

        run_result = self._run_tool_with_optional_repair(
            user_query=user_query,
            agent_route=AgentRoute.QUERY_METADATA,
            tool_call=tool_call,
            repair_llm_role=ModelRole.METADATA_QUERY_REPAIR,
        )

        if isinstance(run_result, ToolError):
            logger.error("Metadata query failed: %s\n%s: %s", user_query, run_result.error_type, run_result.message)
            return "Sorry, something went wrong while querying metadata."

        executed_tool_call, result = run_result
        assert not isinstance(result, SearchResult), "SearchResult should not be returned from metadata query tool"

        return self._interpret_metadata_result(
            user_query=user_query,
            tool_call=executed_tool_call,
            result=result,
        )

    def run_once(self, user_query: str) -> str | SearchResult:
        route = self._route(user_query)

        match route:
            case AgentRoute.SEARCH_PHOTOS:
                return self._run_search_photos_pipeline(user_query)

            case AgentRoute.QUERY_METADATA:
                return self._run_metadata_query_pipeline(user_query)

            case _:
                logger.warning("Unsupported query: %s", user_query)
                return UNSUPPORTED_MESSAGE


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    from lensieve.logging_config import setup_logging
    from lensieve.models.model_manager import get_model_manager

    @hydra.main(version_base=None, config_path="../../../configs", config_name="agent_config")
    def main(cfg: DictConfig) -> None:
        setup_logging(cfg.root, app_name="photo_agent")
        model_manager = get_model_manager(cfg)
        data_store = DataStore(cfg.root)

        agent = PhotoAgent(
            model_manager=model_manager,
            data_store=data_store,
            northern_hemisphere=cfg.agent.northern_hemisphere,
            max_image_results=cfg.agent.tools.search_photos.max_results,
            max_metadata_results=cfg.agent.tools.query_metadata.max_results,
        )

        query = "Dog"
        res = agent.run_once(query)
        if type(res) is str:
            print(res)
        else:
            print(type(res))

    main()
