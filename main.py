import argparse
import logging
import os
from scripts.generate_inputs import generate_inputs
from scripts.generate_outputs import generate_outputs
from core.engines.hf_inference_engine import HFInferenceEngine
from core.factory.engine_factory import create_engine
from core.factory.prompt_formatter_factory import create_prompt_formatter
from utils.io_utils import load_yaml

ENV_CONFIG_PATH = "config/env_config.yaml"
MODEL_CONFIG_PATH = "config/model_config.yaml"
INPUT_PROMPT_TEMPLATE_PATH = "prompt_templates/gen_input_prompt.yaml"
OUTPUT_PROMPT_TEMPLATE_PATH = "prompt_templates/gen_output_prompt.yaml"
GEN_INPUT_FILE_PATH = "data/raw/generated_inputs.jsonl"
GEN_OUTPUT_FILE_PATH = "data/synth/synthetic_data.jsonl"

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Dataset Generator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--num_samples",
        type=int,
        help="Number of synthetic input samples to generate with an LLM. Only used if input_file not provided"
    )
    group.add_argument(
        "--input_file",
        type=str,
        help="Path to JSONL file containing user-prepared questions (e.g., data/sample/simple_qa.jsonl)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=GEN_OUTPUT_FILE_PATH,
        help="Path to save the final synthetic outputs"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="vllm",
        choices=["hf", "vllm", "openai"],
        help="Inference engine to use: 'hf' for HuggingFace, 'vllm' for vLLM, 'openai' for OpenAI API"
    )

    args = parser.parse_args()

    # Set environment variables
    env = load_yaml(ENV_CONFIG_PATH)
    model_config = load_yaml(MODEL_CONFIG_PATH)
    os.environ["HF_TOKEN"] = env["environment"]["HF_TOKEN"]
    os.environ["OPENAI_API_KEY"] = env["environment"]["OPENAI_API_KEY"]

    # Prepare input synthetic data
    if args.input_file:
        logging.info(f"Using user-provided questions from: {args.input_file}")
    else:
        logging.info(f"Generating {args.num_samples} synthetic inputs...")
        # Generate input synthetic data
        input_prompt_config = load_yaml(INPUT_PROMPT_TEMPLATE_PATH)
        input_engine = create_engine(args.engine, model_config["input_model"]["model_id"])

        input_prompt_formatter = create_prompt_formatter(
            formatter_type=model_config["input_model"]["prompt_format"],
            system_context=input_prompt_config["system_context"],
            user_prompt_prefix=input_prompt_config["user_prompt_prefix"]
        )
        generate_inputs(
            engine=input_engine,
            prompt_formatter=input_prompt_formatter,
            num_samples=args.num_samples,
            output_path=GEN_INPUT_FILE_PATH
        )
        input_engine.release()

    # Generate output synthetic data
    logging.info("Transforming inputs into outputs...")
    output_prompt_config = load_yaml(OUTPUT_PROMPT_TEMPLATE_PATH)
    output_engine = create_engine(args.engine, model_config["output_model"]["model_id"])

    output_prompt_formatter = create_prompt_formatter(
        formatter_type=model_config["output_model"]["prompt_format"],
        system_context=output_prompt_config["system_context"],
        user_prompt_prefix=output_prompt_config["user_prompt_prefix"]
    )
    generate_outputs(
        engine=output_engine,
        prompt_formatter=output_prompt_formatter,
        input_path=args.input_file if args.input_file else GEN_INPUT_FILE_PATH,
        output_path=args.output_file
    )
    output_engine.release()
    logging.info(f"Synthetic dataset generation complete. Output saved to {args.output_file}")
