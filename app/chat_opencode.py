from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import SecretStr, Field


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

            # Test request
            model.invoke("test")

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

            model.invoke("test")

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
        return self.client._generate(
            messages,
            stop=stop or self.stop,
            run_manager=run_manager,
            **merged_kwargs,
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
