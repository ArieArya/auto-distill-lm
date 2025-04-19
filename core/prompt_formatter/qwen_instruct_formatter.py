from core.interfaces.prompt_formatter import PromptFormatter

class QwenInstructPromptFormatter(PromptFormatter):
    def __init__(
        self,
        system_context: str = "You are a helpful assistant.",
        user_prompt_prefix: str = ""
    ):
        self.prompt_template = (
            f"<|im_start|>system\n{system_context}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt_prefix}{{input}}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def get_template(self):
        return self.prompt_template
