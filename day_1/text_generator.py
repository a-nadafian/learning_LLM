# Step 1: Make sure you have the necessary libraries installed.
# You can install them by running this command in your terminal:
# pip install transformers torch

# Step 2: Import the required classes and libraries.
# - AutoTokenizer is a smart class that can load the correct tokenizer for any model.
# - AutoModelForCausalLM is for loading models designed for text generation (Causal Language Modeling).
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from pathlib import Path


def download_model_and_tokenizer(model_name):
    """
    Handles the downloading and caching of the model and tokenizer.
    """
    print(f"-> Starting download for model: '{model_name}'")
    print("   The library will show a progress bar in the terminal.")

    # The .from_pretrained() method handles everything automatically:
    # 1. It checks if the model is already downloaded in a local cache.
    # 2. If not, it downloads the necessary files from the Hugging Face Hub.
    # 3. It loads the model and tokenizer from the cache.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Let's find the default cache directory to show where files are typically stored.
    cache_dir = os.path.join(Path.home(), ".cache", "huggingface", "hub")
    print(f"\n-> Download complete! Model files are cached.")
    print(f"   (Typically in a directory like: {cache_dir})")

    return tokenizer, model


def main():
    """
    Main function to run the text generation script.
    """
    print("Step 3: Loading the pretrained model and tokenizer...")

    # We'll use "gpt2", a classic and relatively small model from OpenAI.
    # The string "gpt2" is the model identifier on the Hugging Face Hub.
    model_name = "gpt2"

    # Call the function to handle the download and loading process.
    tokenizer, model = download_model_and_tokenizer(model_name)

    print("\nModel and tokenizer loaded successfully!\n")

    # Step 4: Define a prompt and prepare the input for the model.
    prompt = "The future of artificial intelligence is"
    print(f"Our prompt is: '{prompt}'\n")

    # The tokenizer converts our text prompt into a format the model understands: token IDs.
    # `return_tensors="pt"` tells the tokenizer to return the IDs as a PyTorch tensor.
    print("Step 4a: Encoding the prompt into token IDs...")
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Encoded token IDs: {input_ids}\n")

    # Step 5: Use the model to generate text.
    # We pass the token IDs to the model's generate() method.
    # `max_length` controls the total length of the output text (prompt + generated text).
    # `num_return_sequences` specifies how many different completions to generate.
    print("Step 5: Generating text with the model...")
    outputs = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,  # Helps prevent repetitive phrases
        pad_token_id=tokenizer.eos_token_id  # Suppresses a warning message
    )
    print("Text generation complete!\n")

    # Step 6: Decode the generated token IDs back into human-readable text.
    # The model's output (`outputs`) is also a tensor of token IDs.
    # The tokenizer's decode() method converts these IDs back into a string.
    # `skip_special_tokens=True` removes any special tokens like <|endoftext|>.
    print("Step 6: Decoding the output token IDs back to text...")
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("-" * 20)
    print("Generated Text:")
    print(generated_text)
    print("-" * 20)


if __name__ == "__main__":
    main()
