from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List
from utils.io_utils import load_yaml
from core.interfaces.inference_engine import InferenceEngine
from core.types.metrics import InferenceMetric, save_metrics_to_csv
import torch
import logging
import time
import gc

class HFInferenceEngine(InferenceEngine):
    def __init__(self, model_id):
        self.engine_type = "hf"
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4-bit Quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=hf_token,
            device_map="auto",
            quantization_config=bnb_config
        )

    def run_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 200,
        batch_size: int = 64
    ) -> List[str]:
        logging.info(f"Running inference on {len(prompts)} inputs (batch size = {batch_size})")

        all_outputs = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            start = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    repetition_penalty=1.2,
                    temperature=0.7,
                    top_p=0.9
                )
            duration = time.time() - start

            decoded_outputs = []
            for j, output in enumerate(outputs):
                decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                total_generated_tokens += len(self.tokenizer(decoded).input_ids)

                # Remove prompt prefix
                prompt_text = batch_prompts[j].strip()
                if decoded.startswith(prompt_text):
                    decoded = decoded[len(prompt_text):].strip()
                decoded_outputs.append(decoded)

            all_outputs.extend(decoded_outputs)
            self.save_metrics(batch_size, batch_prompts, decoded_outputs, duration)

        logging.info("Inference completed.")
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
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()