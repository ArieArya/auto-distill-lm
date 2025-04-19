from core.interfaces.prompt_formatter import PromptFormatter

class DefaultPromptFormatter(PromptFormatter):
    def __init__(
        self,
        system_context: str = "You are a helpful assistant.",
        user_prompt_prefix: str = ""
    ):
        self.prompt_template = (
            f"{system_context}\n"
            f"{user_prompt_prefix}\n"
            "{{input}}"
        )

    def get_template(self) -> str:
        return self.prompt_template