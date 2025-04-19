from transformers import AutoTokenizer
from core.interfaces.inference_engine import InferenceEngine
from core.types.metrics import InferenceMetric, save_metrics_to_csv
from utils.io_utils import load_yaml
from vllm import LLM, SamplingParams
from typing import List
import torch
import logging
import time
import gc

class VLLMInferenceEngine(InferenceEngine):
    def __init__(self, model_id):
        self.engine_type = "vllm"
        self.model_id = model_id

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            padding_side='left'
        )

        logging.info(f"Loading vLLM model: {self.model_id}")
        self.engine = LLM(
            model=self.model_id,
            trust_remote_code=True,
            task="generate",
            max_model_len=8192,
            dtype="auto",
            quantization="bitsandbytes"  # set quantization
        )

    def run_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 200,
        batch_size: int = 64
    ) -> List[str]:
        logging.info(f"Running vLLM inference on {len(prompts)} inputs (batch size = {batch_size})")

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id],
            repetition_penalty=1.2,
            temperature=0.7,
            top_p=0.9
        )

        all_outputs = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]

            start = time.time()
            outputs = self.engine.generate(batch_prompts, sampling_params)
            duration = time.time() - start

            decoded_outputs = []
            for out in outputs:
                text = out.outputs[0].text.strip()
                decoded_outputs.append(text)
            all_outputs.extend(decoded_outputs)
            self.save_metrics(batch_size, batch_prompts, decoded_outputs, duration)

        return all_outputs

    def save_metrics(self, batch_size, prompts, decoded_outputs, total_duration, save_path="data/metrics/performance.csv"):
        avg_duration = total_duration / batch_size
        metrics = []

        for prompt, output in zip(prompts, decoded_outputs):
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[-1]
            output_tokens = len(self.tokenizer(output).input_ids)
            total_tokens = prompt_tokens + output_tokens
            throughput = total_tokens / avg_duration if avg_duration > 0 else 0

            metric = InferenceMetric(
                engine=self.get_engine_type(),
                model_id=self.get_model_id(),
                total_tokens=total_tokens,
                duration=round(avg_duration, 4),
                throughput=round(throughput, 2),
            )
            metrics.append(metric)

        save_metrics_to_csv(metrics, save_path)

    def release(self):
        del self.engine
        del self.tokenizer
        gc.collect()
        logging.info("Released vLLM engine memory.")
