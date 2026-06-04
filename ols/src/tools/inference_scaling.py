"""Inference-time scaling via its_hub.

Algorithm-agnostic wrapper that delegates sampling and voting to its_hub.
No LangChain dependency — operates on OpenAI-format messages and tool schemas.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import aiohttp

from its_hub import OpenAICompatibleLanguageModel
from its_hub.algorithms import BestOfN, SelfConsistency

logger = logging.getLogger(__name__)

_MAX_SCALING_RETRIES = 2
_SCALING_RETRY_DELAY = 1.0

_TRANSIENT_AIOHTTP = (
    aiohttp.ClientConnectorError,
    aiohttp.ServerDisconnectedError,
    aiohttp.ServerTimeoutError,
)


def _is_transient(exc: BaseException) -> bool:
    """Return True if the error is transient and worth retrying."""
    if isinstance(exc, ExceptionGroup):
        return all(_is_transient(e) for e in exc.exceptions)
    return isinstance(exc, (*_TRANSIENT_AIOHTTP, TimeoutError, ConnectionError))


def create_algorithm(scaling_config: dict[str, Any]) -> Any:
    """Create an its_hub algorithm instance from config."""
    name = scaling_config.get("algorithm", "self_consistency")
    tool_vote = scaling_config.get("tool_vote", "tool_hierarchical")
    exclude_args = scaling_config.get("exclude_args")

    if name == "self_consistency":
        return SelfConsistency(
            tool_vote=tool_vote,
            exclude_args=exclude_args,
        )

    if name == "best_of_n":
        from its_hub import LLMJudge

        judge_model = scaling_config.get("judge_model")
        if not judge_model:
            raise ValueError("best_of_n requires judge_model in config")
        judge_endpoint = scaling_config.get(
            "judge_endpoint", "https://api.openai.com/v1"
        )
        judge_api_key = scaling_config.get("judge_api_key", "")
        judge_lm = OpenAICompatibleLanguageModel(
            endpoint=judge_endpoint,
            api_key=judge_api_key,
            model_name=judge_model,
        )
        return BestOfN(
            LLMJudge(
                lm=judge_lm,
                judge_prompt=scaling_config.get("judge_criterion", ""),
            )
        )

    raise ValueError(
        f"Unknown algorithm: {name!r}. "
        f"Supported: self_consistency, best_of_n"
    )


def _extract_tool_summary(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract a compact tool call summary from an OpenAI response dict."""
    tool_calls = response.get("tool_calls", [])
    if not tool_calls:
        return []
    return [
        {
            "name": tc.get("function", {}).get("name", ""),
            "arguments": tc.get("function", {}).get("arguments", ""),
        }
        for tc in tool_calls
    ]


def _save_trace(
    trace_dir: str,
    round_num: int,
    algorithm_name: str,
    budget: int,
    full_result: Any,
    selected_response: dict[str, Any],
) -> None:
    """Save a full trace of one scaling invocation to a JSON file."""
    trace_path = Path(trace_dir)
    trace_path.mkdir(parents=True, exist_ok=True)

    trace: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "round": round_num,
        "algorithm": algorithm_name,
        "budget": budget,
        "selected": _extract_tool_summary(selected_response),
    }

    if hasattr(full_result, "responses"):
        trace["candidates"] = [
            _extract_tool_summary(r) for r in full_result.responses
        ]
    if hasattr(full_result, "response_counts"):
        counts = full_result.response_counts
        trace["vote_counts"] = [
            {"signature": str(sig), "count": cnt}
            for sig, cnt in counts.most_common()
        ]
    if hasattr(full_result, "scores"):
        trace["scores"] = full_result.scores
    if hasattr(full_result, "selected_index"):
        trace["selected_index"] = full_result.selected_index
    if hasattr(full_result, "usage") and full_result.usage is not None:
        usage = full_result.usage
        trace["usage"] = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "num_calls": usage.num_calls,
        }

    filename = f"round_{round_num:03d}_{int(time.time())}.json"
    filepath = trace_path / filename
    with open(filepath, "w") as f:
        json.dump(trace, f, indent=2, default=str)

    logger.info("Trace saved to %s", filepath)


async def run_scaling(
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    provider_endpoint: str,
    api_key: str,
    model_name: str,
    scaling_config: dict[str, Any],
    round_num: int = 0,
    trace_dir: str | None = None,
) -> dict[str, Any] | None:
    """Run inference-time scaling and return the winning response.

    Creates an its_hub language model and algorithm from config,
    calls ainfer(), and returns the OpenAI-format message dict.
    """
    budget = scaling_config.get("budget", 1)
    if budget <= 1:
        return None

    algorithm_name = scaling_config.get("algorithm", "self_consistency")
    algorithm = create_algorithm(scaling_config)

    lm = OpenAICompatibleLanguageModel(
        endpoint=provider_endpoint,
        api_key=api_key,
        model_name=model_name,
    )

    effective_trace_dir = trace_dir or os.environ.get("ITS_TRACE_DIR")
    save_traces = effective_trace_dir is not None

    result = None
    try:
        last_error: BaseException | None = None
        for attempt in range(1, _MAX_SCALING_RETRIES + 1):
            try:
                result = await algorithm.ainfer(
                    lm,
                    messages,
                    budget=budget,
                    return_response_only=not save_traces,
                    tools=tool_schemas if tool_schemas else None,
                    tool_choice="auto" if tool_schemas else None,
                )
                last_error = None
                break
            except BaseException as exc:
                last_error = exc
                if not _is_transient(exc):
                    raise
                logger.warning(
                    "Transient error in scaling (attempt %d/%d): %s",
                    attempt,
                    _MAX_SCALING_RETRIES,
                    exc,
                )
                if attempt < _MAX_SCALING_RETRIES:
                    await asyncio.sleep(_SCALING_RETRY_DELAY * attempt)
        if last_error is not None:
            raise last_error
    finally:
        await lm.close()

    if result is None:
        return None

    if save_traces and hasattr(result, "the_one"):
        selected = result.the_one
        _save_trace(
            effective_trace_dir,
            round_num,
            algorithm_name,
            budget,
            result,
            selected,
        )
    else:
        selected = result

    tool_calls = selected.get("tool_calls")
    if not tool_calls:
        logger.info(
            "Inference scaling (%s, budget=%d): no tool calls in winning response",
            algorithm_name,
            budget,
        )
        return None

    logger.info(
        "Inference scaling (%s, budget=%d): %d tool calls: %s",
        algorithm_name,
        budget,
        len(tool_calls),
        [tc.get("function", {}).get("name", "") for tc in tool_calls],
    )

    return selected
