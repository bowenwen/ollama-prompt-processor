# Ollama Prompt Processor

This Python project reads prompts from a CSV file, sends them to a specified Ollama Large Language Model (LLM) via its API, collects the responses, validates them against predefined categories, and saves the results to an output CSV file.

It's designed specifically for tasks like categorizing free-text survey responses but can be adapted for other batch prompting tasks.

## Features

*   Reads prompts from a CSV file (`prompt_id`, `prompt_text`).
*   Interacts with a running Ollama instance API (`/api/generate`).
*   Configurable Ollama client (model name, LLM parameters like temperature, top_k, top_p, context window).
*   Validates LLM responses to check if they match one of the predefined categories.
*   Cleans up response data (basic stripping of whitespace/quotes).
*   Exports results to a CSV file, including raw response, cleaned response, and a `fail_status` flag.
*   Includes configurable delay between API calls.
*   Basic error handling for API connection issues, timeouts, and file I/O.

## Prerequisites

1.  **Python:** Version 3.7 or higher recommended.
2.  **Ollama:** You need a running instance of Ollama. Download and install it from [https://ollama.com/](https://ollama.com/).
3.  **Ollama Model:** You need to have pulled the model you intend to use within Ollama (e.g., `ollama pull llama3`). Ensure the model name used in the script matches an available model in your Ollama instance.
4.  **Dependencies:** Install the required Python package.

## Setup

Ollama is required, please download and install Ollama from https://ollama.com/download.

1.  **Clone the repository (or create the files):**
    ```bash
    git clone <repository_url> # Or manually create the directory structure and files
    cd ollama-prompt-processor
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare your input CSV:**
    *   Ensure you have an input CSV file (like `data/example_prompts.csv`) with columns `prompt_id` and `prompt_text`.
    *   The `prompt_text` should contain the full instruction for the LLM, including the text to process and the desired output format/categories.

5.  **Ensure Ollama is running:**
    *   Start the Ollama application or run `ollama serve` in your terminal.
    *   Verify it's accessible (usually at `http://localhost:11434`).


## Configuration

Configuration parameters can be set using command-line arguments, environment variables, or a `.env` file located in the project root.

The order of precedence is:
1.  Command-line arguments (highest priority)
2.  Environment variables (loaded from `.env` or set manually)
3.  Hardcoded defaults in the script (lowest priority)

**Environment Variables / `.env` File:**

Create a file named `.env` in the project's root directory. You can define the following variables:

```dotenv
OLLAMA_MODEL=gemma:7b
OLLAMA_URL=http://localhost:11434
OLLAMA_TEMPERATURE=0.1
OLLAMA_TOP_K=40
OLLAMA_TOP_P=0.9
OLLAMA_CONTEXT_WINDOW=2048
OLLAMA_TIMEOUT=60
PROCESS_DELAY=0.5
INPUT_CSV=data/example_prompts.csv
OUTPUT_CSV=results/output_responses.csv
```

**Command-line Arguments:**

*   `--input-csv` (Required): Path to the input CSV file.
*   `--output-csv` (Required): Path where the output CSV will be saved.
*   `--model` (Optional): Ollama model name. Overrides `OLLAMA_MODEL` env var. Defaults to `gemma3:12b-it`.
*   `--ollama-url` (Optional): Ollama API URL. Overrides `OLLAMA_URL` env var. Defaults to `http://localhost:11434`.
*   `--temperature` (Optional): LLM temperature. Overrides `OLLAMA_TEMPERATURE` env var. Defaults to `0.1`.
*   `--top-k` (Optional): LLM top-k. Overrides `OLLAMA_TOP_K` env var. Defaults to `40`.
*   `--top-p` (Optional): LLM top-p. Overrides `OLLAMA_TOP_P` env var. Defaults to `0.9`.
*   `--context-window` (Optional): Context window size (`num_ctx`). Overrides `OLLAMA_CONTEXT_WINDOW` env var. Defaults to `2048`.
*   `--timeout` (Optional): API timeout (seconds). Overrides `OLLAMA_TIMEOUT` env var. Defaults to `60`.
*   `--delay` (Optional): Delay between API calls (seconds). Overrides `PROCESS_DELAY` env var. Defaults to `0.5`.


## Usage

Run the `main.py` script. You can rely on `.env` for most settings:

```bash
# Example using .env for model, url, etc.
python main.py \
    --input-csv data/example_prompts.csv \
    --output-csv results/output_responses.csv
```

Or override specific settings via command line:

```bash
# Example overriding the model and temperature from .env/defaults
python main.py \
    --input-csv data/example_prompts.csv \
    --output-csv results/output_responses.csv \
    --model llama3 \
    --temperature 0.5
```


## Output CSV Format

The output CSV file (e.g., `results/output_responses.csv`) will contain the following columns:

*   `prompt_id`: The ID from the input file.
*   `prompt_text`: The full prompt text sent to the LLM.
*   `raw_llm_response`: The raw text response received from the Ollama API. Will contain "API_CALL_FAILED" if the request failed.
*   `cleaned_response`: The response after basic cleaning. If validation against `EXPECTED_CATEGORIES` was successful, this will contain *only* the identified category name. Otherwise, it contains the cleaned-up version of the raw response.
*   `fail_status`: A boolean (`True` or `False`).
    *   `True`: Indicates either the API call failed OR the response validation failed (the response did not cleanly map to one of the `EXPECTED_CATEGORIES`).
    *   `False`: Indicates the API call was successful AND the response was successfully validated and cleaned into one of the expected categories.

## Customization

*   **Expected Categories:** Modify the `EXPECTED_CATEGORIES` list in `main.py` to match the categories relevant to your task and prompts.
*   **Validation Logic:** Enhance the `validate_and_clean_response` function in `utils/validation.py` for more sophisticated parsing or validation if needed (e.g., using regular expressions, handling JSON output from the LLM).
*   **LLM Parameters:** Adjust the default values or command-line options for temperature, top_k, etc., to suit your model and task.
*   **Error Handling:** Add more specific error handling or retry logic in `ollama_client/client.py` or `main.py` as required.




## Project Structure

```
ollama_prompt_processor/
├── ollama_client/      # Handles Ollama API communication
│   ├── __init__.py
│   └── client.py
├── utils/              # Utility functions (e.g., validation)
│   ├── __init__.py
│   └── validation.py
├── data/               # Input data files
│   └── example_prompts.csv
├── results/            # Default output directory (created if needed)
├── main.py             # Main executable script
├── requirements.txt    # Project dependencies
├── README.md           # This file
└── .gitignore          # Git ignore file
```
