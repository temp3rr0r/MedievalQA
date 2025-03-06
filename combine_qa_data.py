import json
import os
import glob
from tqdm import tqdm

def process_book1_format(data):
    """Process book1.json format (SQuAD-like without explicit context)"""
    samples = []
    for article in data["data"]:
        title = article.get("title", "")
        for paragraph in article.get("paragraphs", []):
            # Book1 format doesn't have explicit context field
            # We'll use a concatenation of answer texts as a minimal context
            for qa in paragraph.get("qas", []):
                answers = [ans["text"] for ans in qa.get("answers", [])]
                if not answers:
                    continue
                
                # Use the first answer as implicit context
                context = answers[0] if answers else ""
                
                samples.append({
                    "id": qa.get("id", ""),
                    "title": title,
                    "context": context,
                    "question": qa.get("question", ""),
                    "answers": answers
                })
    return samples

def process_book2_format(data):
    """Process book2.json format (context-QA pairs)"""
    samples = []
    for item in data.get("data", []):
        context = item.get("context", "")
        for qa in item.get("qas", []):
            answers = [ans["text"] for ans in qa.get("answers", [])]
            if not answers:
                continue
            
            samples.append({
                "id": qa.get("id", ""),
                "context": context,
                "question": qa.get("question", ""),
                "answers": answers
            })
    return samples

def process_book3_format(data):
    """Process book3.json format (id, question, multiple answers)"""
    samples = []
    # Check if data is a dict with 'data' key or directly a list
    if isinstance(data, dict) and "data" in data:
        items = data.get("data", [])
    else:
        items = []
    
    for item in items:
        answers = item.get("answers", [])
        # Use the first answer as implicit context
        context = answers[0] if answers else ""
        
        samples.append({
            "id": str(item.get("id", "")),
            "question": item.get("question", ""),
            "answers": answers,
            "context": context
        })
    return samples

def process_list_format(data):
    """Process a list format (books 4-7)"""
    samples = []
    # Ensure data is a list
    if not isinstance(data, list):
        return samples
    
    for i, item in enumerate(data):
        answer = item.get("answer", "")
        samples.append({
            "id": str(i+1),  # Generate ID
            "question": item.get("question", ""),
            "answers": [answer],
            "context": answer  # Use answer as context
        })
    return samples

def process_book8_format(data):
    """Process book8.json format (simple question-answer pairs)"""
    samples = []
    for i, item in enumerate(data.get("data", [])):
        answer = item.get("answer", "")
        samples.append({
            "id": str(i+1),  # Generate ID
            "question": item.get("question", ""),
            "answers": [answer],
            "context": answer  # Use answer as context
        })
    return samples

def fix_book1_json(file_path):
    """Fix the malformed JSON in book1.json"""
    try:
        # Read the file as a string
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix the common JSON syntax error (extra comma before closing brace)
        fixed_content = content.replace("},\n\t\t}", "}\n\t\t}")
        
        # Try to parse the fixed content
        data = json.loads(fixed_content)
        return data
    except Exception as e:
        print(f"Failed to fix {file_path}: {str(e)}")
        return None

def main():
    output_file = "combined_qa_dataset.json"
    all_samples = []
    
    # Get all JSON files in the qa_data directory
    json_files = glob.glob("qa_data/*.json")
    print(f"Found {len(json_files)} JSON files")
    
    for file_path in tqdm(json_files, desc="Processing files"):
        try:
            filename = os.path.basename(file_path)
            
            # Special handling for book1.json which has malformed JSON
            if "book1.json" in filename:
                data = fix_book1_json(file_path)
                if data:
                    samples = process_book1_format(data)
                else:
                    continue  # Skip if can't be fixed
            else:
                # Normal JSON loading
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Determine format and process accordingly
                if "book2.json" in filename:
                    samples = process_book2_format(data)
                elif "book3.json" in filename:
                    samples = process_book3_format(data)
                elif "book4.json" in filename or "book5.json" in filename or "book6.json" in filename or "book7.json" in filename:
                    samples = process_list_format(data)
                elif "book8.json" in filename:
                    samples = process_book8_format(data)
                else:
                    # Default to book3 format which is most generic
                    samples = process_book3_format(data)
            
            print(f"Processed {filename}: {len(samples)} samples")
            all_samples.extend(samples)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Create final HuggingFace format dataset
    hf_dataset = {
        "version": "1.0",
        "data": all_samples
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(hf_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Combined dataset saved to {output_file}")
    print(f"Total samples: {len(all_samples)}")

if __name__ == "__main__":
    main() 