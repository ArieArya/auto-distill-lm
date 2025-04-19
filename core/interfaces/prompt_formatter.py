from abc import ABC, abstractmethod
from typing import List

class PromptFormatter(ABC):
    @abstractmethod
    def get_template(self) -> str:
        pass

    def format_prompt(self, user_prompt) -> str:
        return self.get_template().replace("{{input}}", user_prompt)
