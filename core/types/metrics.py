from dataclasses import dataclass, asdict
from typing import List
import csv
import os
import logging

@dataclass
class InferenceMetric:
    engine: str  # e.g., 'vllm', 'hf', 'openai'
    model_id: str
    total_tokens: int
    duration: float  # seconds
    throughput: float  # tokens/sec

    def to_dict(self):
        return asdict(self)

def save_metrics_to_csv(metrics: List[InferenceMetric], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = list(metrics[0].to_dict().keys())

    file_exists = os.path.isfile(output_path)

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for m in metrics:
            writer.writerow(m.to_dict())

    logging.info(f"Saved {len(metrics)} inference metrics to {output_path}")
