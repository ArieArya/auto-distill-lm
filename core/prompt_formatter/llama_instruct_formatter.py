from core.interfaces.prompt_formatter import PromptFormatter

class LlamaInstructPromptFormatter(PromptFormatter):
    def __init__(
        self,
        system_context: str = "You are a helpful assistant.",
        user_prompt_prefix: str = ""
    ):
        self.prompt_template = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system_context}\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{user_prompt_prefix}\n"
            "{{input}}"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

    def get_template(self):
        return self.prompt_template