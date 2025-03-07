# Medieval Arms and Armor Q&A System

A specialized question-answering system focused on medieval arms, armor, and related topics using a fine-tuned LLaMA 3.2 model.

## Project Overview

This project implements a specialized question-answering system that can respond to queries about medieval weaponry, armor, and historical combat. It leverages a fine-tuned LLaMA 3.2 model specifically trained on a curated dataset of medieval arms and armor information.

## Dataset

The dataset consists of question-answer pairs extracted from various books on medieval arms and armor:

- Located in the `qa_data/` directory
- Contains 8 JSON files (`book1.json` through `book8.json`) with Q&A pairs
- Combined into a single dataset using `combine_qa_data.py`
- Final combined dataset: `combined_qa_dataset.json`

## Model

- Base model: LLaMA 3.2 (3B parameters)
- Fine-tuning process documented in: `sft-and-merge-llama-3-2-medievalarmsandarmor.ipynb`
- Quantized model: `llama-3.2-3b-medieval-arms-and-armor-q4_k_m.gguf` (Q4_K_M quantization)
- Modelfile for ollama: `llama3.2-medieval.Modelfile`

## Setup Instructions

### Prerequisites

- Python 3.8+
- Git LFS (for handling large model files)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/MedievalQA.git
   cd MedievalQA
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt  # Note: requirements.txt needs to be created
   ```

### Using with Ollama (optional)

If you want to run the model using Ollama:

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Create the model using the provided Modelfile:
   ```bash
   ollama create medieval-qa -f llama3.2-medieval.Modelfile
   ```
3. Run the model:
   ```bash
   ollama run medieval-qa
   ```

## Usage

### Interacting with the model

You can ask the model questions about medieval arms, armor, combat techniques, historical context, and related topics. For example:

- "What materials were used to make chain mail?"
- "How heavy was a typical medieval sword?"
- "What was the purpose of a gorget?"
- "How did knights train for combat?"

### Working with the dataset

If you want to work with or extend the dataset:

1. Add new Q&A pairs to individual book files or create new ones in the `qa_data/` directory
2. Run the combination script to update the combined dataset:
   ```bash
   python combine_qa_data.py
   ```

## File Structure

- `qa_data/`: Directory containing individual JSON files with Q&A pairs
- `combined_qa_dataset.json`: Combined dataset used for fine-tuning
- `combine_qa_data.py`: Script to combine individual datasets
- `sft-and-merge-llama-3-2-medievalarmsandarmor.ipynb`: Notebook documenting the fine-tuning process
- `llama-3.2-3b-medieval-arms-and-armor-q4_k_m.gguf`: Quantized model file
- `llama3.2-medieval.Modelfile`: Configuration for using with Ollama


## Acknowledgements

- Meta AI for the base LLaMA 3.2 model