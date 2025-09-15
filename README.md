# Fine-Tuned Models for JSON Generation

This repository contains multiple fine-tuned models designed to generate valid JSON outputs from natural language prompts, following predefined schemas:

## JSON Generation Models

1. **[LLaMa-3.1-8B-Finetuned-for-JSON-Generation]**
2. **[LLaMa-3.2-3B-Finetuned-for-JSON-Generation]**
3. **[Qwen2.5-7B-Finetuned-for-JSON-Generation]**

These models are fine-tuned on the [Salesforce XLAM Function Calling Dataset](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) dataset to enhance their ability to produce structured JSON data from text queries.

## Model Overview

### 1. LLaMa-3.1-8B-Finetuned-for-JSON-Generation
- **Base Model**: [unsloth/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct)
- **Fine-Tuning Dataset**: [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- **Objective**: Convert text prompts or invalid JSON inputs into valid JSON outputs following a predefined schema. Finetuned using 4bit Unsloth quantization.
- **Model Repository**: [LLaMa-3.1-8B-Finetuned-for-JSON-Generation]
- **Notebooks**: llama3.1_8b_finetuned_for_json_generation_unsloth_4bit_instruct.ipynb and llama3.1_8b_finetuned_for_json_generation_unsloth_4bit.ipynb

### 2. LLaMa-3.2-3B-Finetuned-for-JSON-Generation
- **Base Model**: [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- **Fine-Tuning Dataset**: [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- **Objective**: Convert text prompts or invalid JSON inputs into valid JSON outputs following a predefined schema.
- **Model Repository**: [LLaMa-3.2-3B-Finetuned-for-JSON-Generation]

### 3. Qwen2.5-7B-Finetuned-for-JSON-Generation
- **Base Model**: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **Fine-Tuning Dataset**: [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- **Objective**: Convert text prompts or invalid JSON inputs into valid JSON outputs following a predefined schema.
- **Model Repository**: [Qwen2.5-7B-Finetuned-for-JSON-Generation]

## Features

- **Text to JSON Conversion**: The models transform natural language queries into structured JSON objects.
- **Invalid to Valid JSON Correction**: Corrects malformed JSON inputs into valid JSON outputs.
- **Schema Compliance**: All models ensure that the outputs adhere to predefined schemas, ensuring consistency and reliability.

## Usage

### Loading the Fine-Tuned Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model (Replace 'model_huggingface_repo' with the desired model's repository name)
model = AutoModelForCausalLM.from_pretrained(
    "model_huggingface_repo",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("model_huggingface_repo")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Add special tokens if not already present
special_tokens = {
    'additional_special_tokens': [
        '<|begin_of_text|>',
        '<|start_header_id|>',
        '<|end_header_id|>',
        '<|eot_id|>',
        '<|endoftext|>'
    ]
}
num_added_toks = tokenizer.add_special_tokens(special_tokens)
if num_added_toks > 0:
    model.resize_token_embeddings(len(tokenizer))
