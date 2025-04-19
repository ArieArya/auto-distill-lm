from core.prompt_formatter.default_formatter import DefaultPromptFormatter
from core.prompt_formatter.llama_instruct_formatter import LlamaInstructPromptFormatter
from core.prompt_formatter.qwen_instruct_formatter import QwenInstructPromptFormatter
from core.interfaces.prompt_formatter import PromptFormatter

def create_prompt_formatter(
    formatter_type: str,
    system_context: str,
    user_prompt_prefix: str
) -> PromptFormatter:
    formatter_type = formatter_type.lower()

    if formatter_type == "default":
        return DefaultPromptFormatter(system_context, user_prompt_prefix)
    elif formatter_type == "llama-instruct":
        return LlamaInstructPromptFormatter(system_context, user_prompt_prefix)
    elif formatter_type == "qwen-instruct":
        return QwenInstructPromptFormatter(system_context, user_prompt_prefix)
    else:
        raise ValueError(f"Unsupported formatter_type: {formatter_type}. Supported formatter_type: ['default', 'llama-instruct', 'qwen-instruct']")
