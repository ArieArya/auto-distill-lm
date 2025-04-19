import os
import gc
import logging
import torch
from tqdm import tqdm
from core.interfaces.inference_engine import InferenceEngine
from core.interfaces.prompt_formatter import PromptFormatter
from utils.io_utils import load_jsonl, save_jsonl

PROMPT_BATCH_SIZE = 256  # note this is different from the inference batch size

def generate_outputs(
    engine: InferenceEngine,
    prompt_formatter: PromptFormatter,
    input_path: str,
    output_path: str
):
    logging.info(f"Loading inputs from: {input_path}")
    inputs = load_jsonl(input_path)
    transformed = []
    for i in tqdm(range(0, len(inputs), PROMPT_BATCH_SIZE), desc="Transforming inputs"):
        input_batch = inputs[i:i+PROMPT_BATCH_SIZE]
        prompt_batch = [prompt_formatter.format_prompt(item["input"]) for item in input_batch]
        outputs = engine.run_batch(prompt_batch, max_new_tokens=256, batch_size=256)

        for item, prompt, output in zip(input_batch, prompt_batch, outputs):
            transformed.append({
                "id": item["id"],
                "input": item["input"],  # intentionally hide the prompts
                "output": output
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_jsonl(transformed, output_path)
    logging.info(f"Transformed {len(inputs)} input samples into outputs.")
    logging.info(f"Saved output to: {output_path}")
