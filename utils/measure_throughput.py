import time
from core.inference_engine import InferenceEngine

if __name__ == "__main__":
    output_config_path = "./config/output_config.yaml"
    engine = InferenceEngine(output_config_path)
    prompt = "Explain quantum computing in simple terms."

    print("Measuring throughput...")
    start = time.time()
    output = engine.run_batch([prompt], max_new_tokens=max_tokens)[0]
    end = time.time()

    tokens = len(engine.tokenizer(output).input_ids)
    duration = end - start
    throughput = tokens / duration

    print("--- Inference Summary ---")
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")
    print(f"Number of Tokens: {tokens}")
    print(f"Time Taken: {duration:.2f} seconds")
    print(f"Throughput: {throughput:.2f} tokens/sec")