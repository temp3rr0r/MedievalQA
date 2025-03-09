#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to upload the combined_qa_dataset.json file to the Hugging Face dataset
'madks/medieval-qa-dataset' in Parquet format.
"""

import json
import os
from datasets import Dataset
from huggingface_hub import login
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Upload JSON dataset to Hugging Face in Parquet format")
    parser.add_argument("--input", default="combined_qa_dataset.json", help="Path to the input JSON file")
    parser.add_argument("--repo", default="madks/medieval-qa-dataset", help="Hugging Face repository ID")
    parser.add_argument("--token", help="Hugging Face API token (will prompt if not provided)")
    parser.add_argument("--version", default="default", help="Dataset version/config name")
    args = parser.parse_args()
    
    # Load the JSON data
    print(f"Loading the dataset from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    
    # Convert to a more straightforward format for Hugging Face datasets
    print("Converting data to dataset format...")
    examples = []
    for item in qa_data["data"]:
        example = {
            "question": item["question"],
            "context": item.get("context", ""),  # Some may not have context
            "answers": item["answers"][0] if item["answers"] else ""  # Taking the first answer
        }
        examples.append(example)
    
    # Create a Hugging Face Dataset
    print(f"Creating dataset with {len(examples)} examples...")
    dataset = Dataset.from_pandas(pd.DataFrame(examples))
    
    # Login to Hugging Face
    huggingface_token = args.token
    if not huggingface_token:
        print("Please enter your Hugging Face token when prompted...")
        huggingface_token = input("Hugging Face token: ")
    
    login(token=huggingface_token)
    
    # Push to the Hub in Parquet format
    print(f"Uploading to Hugging Face Hub at {args.repo}...")
    dataset.push_to_hub(
        repo_id=args.repo,
        config_name=args.version,
        token=huggingface_token,
        private=False,
    )
    
    print(f"Upload complete! The dataset is now available at https://huggingface.co/datasets/{args.repo}")

if __name__ == "__main__":
    main() 