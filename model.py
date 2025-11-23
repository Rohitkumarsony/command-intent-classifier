from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from load_datasets import dataset
import torch

# Model Name
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token  # Required for LLaMA

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Preprocessing
def preprocess_function(examples):
    text_batch = [
        prompt + "\n" + completion
        for prompt, completion in zip(examples["prompt"], examples["completion"])
    ]

    tokens = tokenizer(
        text_batch,
        truncation=True,
        padding="max_length",
        max_length=512
    )

    # LLaMA requires labels = input_ids
    tokens["labels"] = tokens["input_ids"].copy()

    return tokens

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_total_limit=2,
    fp16=True,  # Faster & memory efficient
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save final model
model.save_pretrained("./fine_tuned_llama")
tokenizer.save_pretrained("./fine_tuned_llama")

print("Fine-tuning complete! Model saved at ./fine_tuned_llama")
