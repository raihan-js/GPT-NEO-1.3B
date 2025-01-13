import torch
from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Check GPU availability
print("CUDA available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device for Flask app:", device)

# Define the model directory
model_path = "./models/blackgpt"  # Update with the path where your Moxin 7B model is saved

try:
    # Load fine-tuned Moxin 7B model and tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"  # Use accelerate to handle device mapping
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Explicitly set pad_token_id to eos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create the pipeline for text generation
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model loading failed: {e}")
    # Fallback: Dummy generator for responses
    def dummy_generator(prompt, **kwargs):
        return [{"generated_text": prompt + " [Dummy response because model is not loaded.]"}]
    generator = dummy_generator

@app.route('/')
def index():
    return render_template('chat.html')  # Ensure chat.html exists in the templates folder


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('user_input', '')
    print(f"Received user input: {user_input}")  # Debug log

    prompt = f"User: {user_input}\nAI:"
    try:
        response = generator(
            prompt,
            max_length=350,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            truncation=True
        )
        ai_text = response[0]['generated_text'].split("AI:")[-1].strip()
        print(f"AI response: {ai_text}")  # Debug log
        return jsonify({"ai_response": ai_text})
    except Exception as e:
        print(f"Error during response generation: {e}")  # Debug log
        return jsonify({"error": "Error retrieving response"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
