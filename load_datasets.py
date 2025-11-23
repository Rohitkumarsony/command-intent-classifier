import json
from datasets import Dataset

def load_jsonl_dataset(filepath="commands.jsonl"):
    prompts = []
    completions = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            # Ensure keys exist
            prompt = obj.get("prompt", "")
            completion = obj.get("completion", "")

            prompts.append(prompt)
            completions.append(completion)

    dataset_dict = {
        "prompt": prompts,
        "completion": completions
    }

    return Dataset.from_dict(dataset_dict)


# Load dataset
dataset = load_jsonl_dataset()
