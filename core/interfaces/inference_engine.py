from abc import ABC, abstractmethod
from typing import List

class InferenceEngine(ABC):
    @abstractmethod
    def run_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 200,
        batch_size: int = 64
    ) -> List[str]:
        pass

    @abstractmethod
    def release(self):
        pass

    def get_model_id(self):
        return self.model_id

    def get_engine_type(self):
        return self.engine_type