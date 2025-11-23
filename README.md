## COMMAND INTENT CLASSIFIER

PROJECT OVERVIEW
----------------------------------------
The Command Intent Classifier is a transformer-based machine learning system designed to identify the intent behind Linux commands or natural language inputs. It fine-tunes the LLaMA 3.1 8B model to classify text into predefined command categories.

This project is ideal for:
• AI terminal assistants
• DevOps automation
• Shell command recommendations
• ChatOps tools
• Custom command understanding systems


PROJECT STRUCTURE
----------------------------------------
command-intent-classifier/
    commands.jsonl         -> Dataset containing command prompts and completions
    inference.py          -> Runs inference on the fine-tuned model
    load_datasets.py      -> Loads and processes the dataset
    model.py              -> Fine-tuning script for the LLaMA model
    README.md             -> Markdown Readme (GitHub)
    requirements.txt      -> Required Python packages

Generated after training:
    fine_tuned_llama/     -> Saved fine-tuned model
    results/              -> Training checkpoints
    logs/                 -> Training logs


FEATURES
----------------------------------------
• Transformer-based command intent classification  
• Uses LLaMA 3.1 8B Instruct model  
• Supports custom prompt–completion training format  
• Mini-batch transformer training using HuggingFace Trainer  
• Clean inference interface to query the trained model  
• Easily extendable with more command categories  


HOW TRAINING WORKS
----------------------------------------
The model uses prompt + completion style training.

Example training pair:
Prompt: "Show only hidden files"
Completion: "intent: list_hidden_files"

Training configuration:
• Model: meta-llama/Llama-3.1-8B-Instruct
• Max length: 512 tokens
• Batch size: 2
• Learning rate: 2e-5
• Epochs: 3
• Save strategy: per epoch


FINE-TUNING THE MODEL
----------------------------------------

## REQUIREMENTS
Install dependencies using:
```
pip install -r requirements.txt
```
----------------------------------------
To start training:

1. Make sure your dataset is inside commands.txt
2. Run the training script:
```
python3 model.py
```
This will:
• Load dataset  
• Tokenize  
• Fine-tune LLaMA  
• Save model to fine_tuned_llama/  


RUNNING INFERENCE
----------------------------------------
After the model is fine-tuned, run:
```
python3 inference.py
```
Then type any Linux command or natural language instruction.

Example:
Input: "How do I check disk usage?"
Output: "intent: disk_usage"


FOLDER DESCRIPTIONS
----------------------------------------
commands.jsonl
    Contains your training data in prompt + completion format.

inference.py
    Loads fine-tuned model and returns predicted intent.

load_datasets.py
    Reads commands.txt and converts it into a HuggingFace Dataset.

model.py
    Fine-tunes the LLaMA model using TrainingArguments and Trainer API.

README.md
    GitHub-friendly version of project documentation.

requirements.txt
    Contains required libraries.




