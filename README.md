# AutoDistill

This project provides a flexible framework for generating synthetic datasets using large language models as teacher models for distillation or downstream fine-tuning. It enables batch prompting and inferencing, flexible prompt formatting, and metrics logging — making it easy to distill knowledge from stronger models into smaller ones.

The framework supports a number of inference backends, including Hugging Face Transformers, OpenAI APIs, and vLLM servers. Users can use any Hugging Face-compatible models and configure prompt formatting logic as needed.

---

## Key Features

-   Customizable prompt template formatting (supports special formatting for instruct-based / chat-based models)
-   Flexible inference engines: Hugging Face Transformers, OpenAI API, and vLLM
-   Logs token counts, duration, and throughput for each prompt for experimentation / analysis
-   Outputs JSONL synthetic datasets suitable for supervised finetuning / training
-   Easily extendable: define new prompt formatters, models, or output handlers

---

## Hardware Recommendations

We strongly recommended to use GPU-enabled instances for high-throughput generation. Suitable cloud-providers include:

-   Google Cloud Platform (GCP)
-   AWS EC2 (with A10G, A100, or T4 GPUs)
-   Lambda Labs
-   Vast AI

Quantized models (e.g., 4-bit AWQ or bitsandbytes) are supported via vLLM to reduce memory usage.

---

## Environment Setup

Clone the repository:

```bash
git clone git@github.com:ArieArya/auto-distill-lm.git
```

Install dependencies:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

If you require a Hugging Face token (to access gated models) or OpenAI API key, you must set them up in `config/env_config.yaml`.

Further, you can choose the teacher models for generating synthetic data in `config/model_config.yaml`. Note that `input_model` refers to the model used to generate input synthetic data, whilst `output_model` refers to the model used to generate the output synthetic data. `prompt_format` can be used to wrap your prompts with special tokens as required by some model (see [here](https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/) for an example of the Llama 3 prompt format).

Lastly, you can define the prompt used for your specific task under `prompt_templates`. Here, you must provide a `system_context` if working with chat-based teacher models, and the `user_prompt_prefix` for any prefix string that comes before your actual user prompt. The example used in this repository is for customer review summarization.

---

## Example Usage

### Usage 1: End-to-end Synthetic Data Generation

In the case where you want your teacher model to also generate synthetic input data, you must provide both the `prompt_templates/gen_input_prompt.yaml` and the `prompt_templates/gen_output_prompt.yaml`.

The first prompt will be used to generate synthetic input data. For example, if you want to generate synthetic training data for text summarization, you need to first generate synthetic input data (i.e., the long texts to summarize), then generate synthetic output data (i.e., the summary of those texts).

The below command will generate 1000 synthetic data points for the task you define.

```
python main.py \
	--num_samples 1000 \
	--engine vllm \
	--output_file {output-file-path}
```

### Usage 2: Synthetic Data Generation for a given Input Data

If you have the input data and would like to generate synthetic output data, you must provide the `--input_file` flag. The input file must be a JSONL file in the following format:

```
{"id": "0", "input": "What are the key differences between supervised and unsupervised learning?"}
{"id": "1", "input": "Explain the concept of attention in transformer models."}
...
```

See examples of valid input files in `data/sample/sample_qa.jsonl`. As before, you must provide the `prompt_templates/gen_output_prompt.yaml` to instruct the teacher model on how to generate the synthetic data. Next, you can run the following:

```
python main.py \
	--engine vllm \
	--input_file {input-file-path} \
	--output_file {output-file-path}
```

The generated output will be a JSONL file in the form:

```
{"id": "0", "input": "What are the key differences between supervised and unsupervised learning?", "output": "..."}
```

The "output" key contains the generated synthetic data.

---

## Performance Logging

Each inference batch logs the following per-prompt metrics:

-   Prompt text
-   Model output
-   Number of tokens (prompt, output, total)
-   Inference duration (seconds)
-   Throughput (tokens per second)
-   Engine type and model ID

These are saved to a CSV file (e.g., `data/metrics/performance.csv`).

---

## Comet QE Score

If you want to compute the [Comet QE score](https://arxiv.org/abs/2210.15696) of your generation outputs, you can run:

```
python3 utils/compute_comet_qe.py
```

Comet QE is initially created as a reference-free metric for machine translation, but may be used to some degree for analyzing the fluency of the generation output. Make sure to change the input and output file paths in this script to the desired paths.

---

## Project Deliverables

Please find the project deliverables below:

-   [Project Proposal](https://drive.google.com/file/d/1NZWhMYj5ATUxrVXKDAbJgNPagZ0uXH-I/view?usp=sharing)
-   [Progress Report](https://drive.google.com/file/d/1sSlcaWiiFDdSple8qRkneX3nyi1rnNGJ/view?usp=sharing)
-   [Final Report](https://drive.google.com/file/d/1LYuqSoN2TstpqEKvX4mrSvloDmT49D6l/view?usp=sharing)
