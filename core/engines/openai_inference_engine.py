import openai
import logging
import time
import gc
from typing import List
from core.interfaces.inference_engine import InferenceEngine
from core.types.metrics import InferenceMetric, save_metrics_to_csv
from utils.io_utils import load_yaml

class OpenAIInferenceEngine(InferenceEngine):
    def __init__(self, model_id):
        self.engine_type = "openai"
        self.model_id = model_id
        self.client = openai.OpenAI()

    def run_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 200,
        batch_size: int = 1  # OpenAI API typically doesn't support true batching via one call
    ) -> List[str]:
        logging.info(f"Running OpenAI inference on {len(prompts)} inputs")

        results = []
        for prompt in prompts:
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                )
                duration = time.time() - start
                total_tokens = response.usage.total_tokens
                result = response.choices[0].message.content.strip()
                results.append(result)
                self.save_metrics(total_tokens, duration)
            except Exception as e:
                logging.warning(f"OpenAI API call failed: {e}")
                results.append("")
        return results

    def save_metrics(self, total_tokens, duration, save_path="data/metrics/performance.csv"):
        throughput = total_tokens / duration if duration > 0 else 0
        metric = InferenceMetric(
            engine=self.get_engine_type(),
            model_id=self.get_model_id(),
            total_tokens=total_tokens,
            duration=round(duration, 4),
            throughput=round(throughput, 2),
        )
        save_metrics_to_csv([metric], save_path)

    def release(self):
        gc.collect()