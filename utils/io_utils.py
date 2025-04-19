import json
import yaml

def save_jsonl(data, output_path):
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def load_jsonl(input_path):
    with open(input_path, 'r') as f:
        return [json.loads(line) for line in f]

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)