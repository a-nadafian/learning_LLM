# Day 1: Your First Text Generation Script

Welcome to Day 1 of the 10-day LLM learning series! Today, we'll dive right in by running a simple Python script to generate text using a pretrained language model.

## What is this script?

The `text_generator.py` script is a beginner-friendly example that demonstrates the core workflow of using a powerful, pretrained Language Model (LLM) from the Hugging Face ecosystem.

Here’s what it does, step-by-step:
1.  **Loads a Pretrained Model**: It automatically downloads and loads "gpt2", a well-known model developed by OpenAI.
2.  **Prepares a Prompt**: It defines a starting phrase: `"The future of artificial intelligence is"`.
3.  **Generates Text**: It uses the model to continue writing based on the prompt.
4.  **Prints the Output**: It decodes the model's output and prints the final, human-readable text to your terminal.

## Prerequisites

Before running the script, you need to install the necessary Python libraries. The required packages are listed in the `requirements.txt` file at the root of this repository.

You can install them using pip:
```bash
pip install -r ../requirements.txt
```
*(Note: You need to run this command from the `day_1` directory, which is why we use `../` to go up one level to find the file.)*

## How to Run the Script

Once the prerequisites are installed, you can run the script from your terminal. Make sure you are in the `day_1` directory.

```bash
python text_generator.py
```

The first time you run it, you will see a progress bar as the script downloads the GPT-2 model files (around 500MB). These files are cached locally, so subsequent runs will be much faster.

## Expected Output

You will see a detailed, step-by-step log of what the script is doing. The final output will look something like this (the generated text will vary slightly each time):

```
--------------------
Generated Text:
The future of artificial intelligence is not in the hands of a few people. It is in the hands of the people who are creating it.
--------------------
```

## Key Concepts Introduced

This script introduces you to the fundamental building blocks of working with LLMs:

-   **Model**: The "brain" that has learned patterns from vast amounts of text. In this case, we use `gpt2`.
-   **Tokenizer**: A tool that translates human-readable text into a numerical format (tokens) that the model can understand, and vice-versa.
-   **Prompt**: The initial text you provide to the model to guide its output.
-   **Text Generation**: The process of the model predicting the most likely sequence of tokens to follow a prompt.

Congratulations on running your first LLM script! In the next days, we will explore these concepts in much more detail.
