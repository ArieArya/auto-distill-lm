import os
import json
from comet import download_model, load_from_checkpoint
from tqdm import tqdm

INPUT_FILE = "data/sample_outputs/en_fr_translation.jsonl"
OUTPUT_FILE = "data/comet_qe_score/comet_score.jsonl

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Load COMET-QE model
model_path = download_model("wmt21-comet-qe-mqm")
model = load_from_checkpoint(model_path)

# Load data
data = []
ids = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        data.append({"src": obj["input"], "mt": obj["output"]})
        ids.append(obj["id"])

# Compute scores
results = model.predict(data, batch_size=64)

# Save to new JSONL
with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    for i, score in zip(ids, results["scores"]):
        json.dump({"id": i, "comet-qa-score": score}, out_f)
        out_f.write("\n")

print(f"Saved {len(ids)} scores to {OUTPUT_FILE}")
