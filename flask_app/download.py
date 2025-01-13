from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "moxin-org/moxin-llm-7b"
model_name = "EleutherAI/gpt-neo-1.3B"
model_path = "./models/neo"  # Specify your local model path

print("Downloading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(model_path)

print(f"Model and tokenizer saved to {model_path}.")
