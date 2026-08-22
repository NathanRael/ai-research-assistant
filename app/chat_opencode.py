import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import SecretStr, Field


logger = logging.getLogger(__name__)

# Safe optional imports for transient-error detection.
try:
    import openai
except Exception:
    openai = None  # type: ignore[assignment]

try:
    import httpx
except Exception:
    httpx = None  # type: ignore[assignment]

try:
    import requests
except Exception:
    requests = None  # type: ignore[assignment]


_MAX_RETRIES = 3
_BACKOFF_SECONDS = [1, 2, 4]

T = TypeVar("T")


class OpenCodeModel(str, Enum):
    MINIMAX_M3 = "minimax-m3"
    MINIMAX_M27 = "minimax-m2.7"
    MINIMAX_M25 = "minimax-m2.5"

    KIMI_K3 = "kimi-k3"
    KIMI_K27_CODE = "kimi-k2.7-code"
    KIMI_K26 = "kimi-k2.6"
    KIMI_K25 = "kimi-k2.5"

    GLM_52 = "glm-5.2"
    GLM_51 = "glm-5.1"
    GLM_5 = "glm-5"

    DEEPSEEK_V4_PRO = "deepseek-v4-pro"
    DEEPSEEK_V4_FLASH = "deepseek-v4-flash"

    QWEN_37_MAX = "qwen3.7-max"
    QWEN_38_MAX = "qwen3.8-max"
    QWEN_37_PLUS = "qwen3.7-plus"
    QWEN_36_PLUS = "qwen3.6-plus"
    QWEN_35_PLUS = "qwen3.5-plus"

    MIMO_V2_PRO = "mimo-v2-pro"
    MIMO_V2_OMNI = "mimo-v2-omni"
    MIMO_V25_PRO = "mimo-v2.5-pro"
    MIMO_V25 = "mimo-v2.5"

    HY3 = "hy3"
    HY3_PREVIEW = "hy3-preview"

    GPT_56_LUNA = "gpt-5.6-luna"
    GROK_45 = "grok-4.5"


Provider = Literal["openai", "anthropic"]


def _is_retryable_error(exc: Exception) -> bool:
    """Return True if *exc* represents a transient provider failure worth retrying."""
    if openai is not None:
        if isinstance(exc, (openai.InternalServerError, openai.RateLimitError)):
            return True
        if isinstance(exc, openai.APIStatusError):
            code = exc.status_code
            return code >= 500 or code == 429

    if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code >= 500 or code == 429

    if requests is not None and isinstance(exc, requests.exceptions.HTTPError):
        response = getattr(exc, "response", None)
        if response is not None:
            code = response.status_code
            return code >= 500 or code == 429

    return False


def _error_context(exc: Exception) -> Dict[str, Any]:
    """Return non-sensitive diagnostic fields for logging."""
    ctx: Dict[str, Any] = {"error": exc.__class__.__name__}
    status: Optional[int] = None

    if openai is not None and isinstance(exc, openai.APIStatusError):
        status = exc.status_code
    elif httpx is not None and isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
    elif requests is not None and isinstance(exc, requests.exceptions.HTTPError):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)

    if status is not None:
        ctx["status_code"] = status
    return ctx


def _retry_with_backoff(
    operation: Callable[[], T],
    operation_name: str,
    model_name: str,
) -> T:
    """Execute *operation*, retrying only transient provider failures.

    Retries up to ``_MAX_RETRIES`` times with exponential backoff. The last
    encountered exception is re-raised on final failure so callers see the
    original (or most recent) error.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return operation()
        except Exception as exc:
            last_exc = exc
            if attempt == _MAX_RETRIES or not _is_retryable_error(exc):
                break
            logger.debug(
                "Retryable %s failure for model %s on attempt %d/%d: %s",
                operation_name,
                model_name,
                attempt + 1,
                _MAX_RETRIES,
                _error_context(exc),
            )
            time.sleep(_BACKOFF_SECONDS[attempt])

    if last_exc is not None:
        logger.debug(
            "Final %s failure for model %s after %d attempts: %s",
            operation_name,
            model_name,
            _MAX_RETRIES + 1,
            _error_context(last_exc),
        )
        raise last_exc

    # Defensive fallback; should never be reached.
    raise RuntimeError(f"{_retry_with_backoff.__name__} exited without a result")


class ChatOpenCode(BaseChatModel):
    model_name: str
    client: BaseChatModel = Field(default=None, exclude=True)
    base_url: str

    api_key: SecretStr | str = Field(default=None, exclude=True)
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    streaming: bool = False
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def __init__(
            self,
            api_key: SecretStr | str,
            model_name: str,
            base_url: str = "https://opencode.ai/zen/go/v1",
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            top_p: Optional[float] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            stop: Optional[List[str]] = None,
            streaming: bool = False,
            model_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ):
        if isinstance(api_key, SecretStr):
            api_key = api_key.get_secret_value()

        model_kwargs = model_kwargs or {}

        client = self._detect_provider(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            streaming=streaming,
            model_kwargs=model_kwargs,
        )

        super().__init__(
            model_name=model_name,
            client=client,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            streaming=streaming,
            model_kwargs=model_kwargs,
            **kwargs
        )

    def _detect_provider(
            self,
            api_key: str,
            model_name: str,
            base_url: str,
            temperature: float,
            max_tokens: Optional[int],
            top_p: Optional[float],
            frequency_penalty: Optional[float],
            presence_penalty: Optional[float],
            stop: Optional[List[str]],
            streaming: bool,
            model_kwargs: Dict[str, Any],
    ) -> BaseChatModel:

        errors = []

        common_openai = {
            "model": model_name,
            "api_key": api_key,
            "base_url": base_url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "streaming": streaming,
            "model_kwargs": model_kwargs,
        }

        # Try OpenAI compatible API
        try:
            model = ChatOpenAI(**{k: v for k, v in common_openai.items() if v is not None})

            # Minimal test invocation; avoid passing a raw string that some limited
            # ChatOpenAI implementations may interpret as a structured-output request.
            _retry_with_backoff(
                operation=lambda: model.invoke([HumanMessage(content="hi")]),
                operation_name="provider detection (openai)",
                model_name=model_name,
            )

            return model

        except Exception as e:
            errors.append(
                f"OpenAI failed: {e}"
            )

        common_anthropic = {
            "model": model_name,
            "api_key": api_key,
            "base_url": base_url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stop": stop,
            "streaming": streaming,
        }

        # Try Anthropic compatible API
        try:
            model = ChatAnthropic(**{k: v for k, v in common_anthropic.items() if v is not None})

            _retry_with_backoff(
                operation=lambda: model.invoke([HumanMessage(content="hi")]),
                operation_name="provider detection (anthropic)",
                model_name=model_name,
            )

            return model

        except Exception as e:
            errors.append(
                f"Anthropic failed: {e}"
            )

        raise RuntimeError(
            "Could not detect provider\n"
            + "\n".join(errors)
        )

    @property
    def _llm_type(self) -> str:
        return "opencode"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        merged_kwargs = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            **self.model_kwargs,
            **kwargs,
        }
        merged_kwargs = {k: v for k, v in merged_kwargs.items() if v is not None}

        def _do_generate() -> ChatResult:
            return self.client._generate(
                messages,
                stop=stop or self.stop,
                run_manager=run_manager,
                **merged_kwargs,
            )

        return _retry_with_backoff(
            operation=_do_generate,
            operation_name="generate",
            model_name=self.model_name,
        )

    def bind_tools(
        self,
        tools: list,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        return self.client.bind_tools(tools, tool_choice=tool_choice, **kwargs)

    def with_structured_output(
        self,
        schema: Any,
        **kwargs: Any,
    ) -> BaseChatModel:
        return self.client.with_structured_output(schema, **kwargs)
