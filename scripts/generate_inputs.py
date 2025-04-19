import os
import gc
import logging
import torch
from core.interfaces.inference_engine import InferenceEngine
from core.interfaces.prompt_formatter import PromptFormatter
from utils.io_utils import save_jsonl
from tqdm import tqdm

PROMPT_BATCH_SIZE = 256  # note this is separate from the inference batch size

def generate_inputs(
    engine: InferenceEngine,
    prompt_formatter: PromptFormatter,
    num_samples: int,
    output_path: str,
):
    prompt = prompt_formatter.format_prompt("")  # input pipeline has no custom user inputs
    prompts = [prompt] * num_samples

    results = []
    for i in tqdm(range(0, num_samples, PROMPT_BATCH_SIZE), desc="Generating inputs"):
        prompt_batch = prompts[i:i+PROMPT_BATCH_SIZE]
        decoded_batch = engine.run_batch(prompt_batch, max_new_tokens=1024, batch_size=256)
        for j, decoded in enumerate(decoded_batch):
            results.append({"id": i + j, "input": decoded})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_jsonl(results, output_path)
    logging.info(f"Generated {num_samples} input samples.")
    logging.info(f"Saved samples to: {output_path}")