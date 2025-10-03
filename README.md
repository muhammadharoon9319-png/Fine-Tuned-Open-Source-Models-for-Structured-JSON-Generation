# **Fine-Tuned Open Source Models for Structured JSON Generation**

This repository features a curated collection of fine-tuned open-source Large Language Models (LLMs) specifically optimized for reliable **structured JSON generation** from natural language prompts.

These models are rigorously trained to handle both complex text queries and the correction of malformed JSON, ensuring outputs strictly adhere to predefined schemas. The fine-tuning process utilized the **Salesforce XLAM Function Calling Dataset**, making them highly reliable for function-calling, API integration, and data validation pipelines.

## **✨ Core Features**

The fine-tuned models offer enhanced reliability for downstream applications:

* **Natural Language → JSON**: Directly transforms free-text instructions and complex queries into valid JSON objects.  
* **Malformed JSON Correction**: Identifies and automatically fixes invalid or malformed JSON input, guaranteeing that all outputs are schema-valid.  
* **Schema Enforcement**: Every generated JSON output strictly adheres to the required data schema, vastly improving consistency and integration with external systems.  
* **Open Source & Extendable**: Built upon openly available base models, allowing for further adaptation to custom schemas or specific domains.

## **🚀 Available Models**

The following models are optimized for function calling and structured output:

### **1\. LLaMa-3.1-8B-Finetuned-for-JSON-Generation**

| Detail | Value |
| :---- | :---- |
| **Base Model** | [unsloth/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct) |
| **Dataset** | [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) |
| **Optimization** | Convert text prompts or invalid JSON into valid, schema-compliant JSON. |
| **Training Details** | Fine-tuned with 4-bit Unsloth quantization for efficiency. |
| **Artifacts** | Includes training notebooks: llama3.1\_8b\_finetuned\_for\_json\_generation\_unsloth\_4bit\_instruct.ipynb, llama3.1\_8b\_finetuned\_for\_json\_generation\_unsloth\_4bit.ipynb |

### **2\. LLaMa-3.2-3B-Finetuned-for-JSON-Generation**

| Detail | Value |
| :---- | :---- |
| **Base Model** | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| **Dataset** | [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) |
| **Optimization** | Generate valid, schema-aligned JSON outputs from natural language. |

### **3\. Qwen2.5-7B-Finetuned-for-JSON-Generation**

| Detail | Value |
| :---- | :---- |
| **Base Model** | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| **Dataset** | [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) |
| **Optimization** | Robust handling of free-text queries and malformed JSON, returning consistent schema-valid JSON. |

## **🧑‍💻 Usage Example**

Here’s how to load and configure one of the fine-tuned models using the Hugging Face Transformers library for immediate use:

from transformers import AutoModelForCausalLM, AutoTokenizer  
import torch

\# Replace with the desired fine-tuned model repository name  
model\_name \= "model\_huggingface\_repo" 

\# Load the fine-tuned model  
model \= AutoModelForCausalLM.from\_pretrained(  
    model\_name,  
    torch\_dtype=torch.float16,  
    device\_map="auto"  
)

\# Load and configure the tokenizer  
tokenizer \= AutoTokenizer.from\_pretrained(model\_name)  
tokenizer.pad\_token \= tokenizer.eos\_token  
tokenizer.padding\_side \= "left"

\# Add special tokens required by certain models (like Llama-3.1)  
special\_tokens \= {  
    'additional\_special\_tokens': \[  
        '\<|begin\_of\_text|\>',  
        '\<|start\_header\_id|\>',  
        '\<|end\_header\_id|\>',  
        '\<|eot\_id|\>',  
        '\<|endoftext|\>'  
    \]  
}  
num\_added\_toks \= tokenizer.add\_special\_tokens(special\_tokens)

\# Resize model embeddings if new tokens were added  
if num\_added\_toks \> 0:  
    model.resize\_token\_embeddings(len(tokenizer))

\# You can now use the 'model' and 'tokenizer' for JSON generation tasks.  
