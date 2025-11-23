# ================================
# LOAD FINE-TUNED MODEL & INFER
# ================================
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load fine-tuned model
model_path = "./fine_tuned_llama"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()   # Inference mode

# Important for LLaMA
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token


# Function to ask the model anything
def generate_response(prompt, max_new_tokens=200):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate output
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the response
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return response


# ================================
# Example Usage
# ================================
if __name__ == "__main__":
    question = "Explain Docker in simple words."
    answer = generate_response(question)
    print("Model Response:\n", answer)
