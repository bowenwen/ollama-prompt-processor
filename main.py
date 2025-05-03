# main.py

import csv
import argparse
import logging
import time
import os
import sys  # <-- Import sys for exiting gracefully
import requests  # <-- Import requests
import json  # <-- Import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv

# Import local modules
from ollama_client.client import OllamaClient
from utils.validation import validate_and_clean_response

# Load environment variables from .env file FIRST
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define the expected categories for validation
EXPECTED_CATEGORIES = [
    "Positive Experience",
    "Negative Experience",
    "Neutral Feedback",
    "Feature Request",
    "Bug Report",
    "Other",
]

# Define the new default model name
DEFAULT_MODEL_NAME = (
    "gemma:7b"  # Changed back for example, user can override via .env or CLI
)


# --- Helper Functions for Model Management ---


def check_model_availability(
    model_name: str, ollama_url: str
) -> Tuple[bool, Optional[str]]:
    """
    Checks if the specified Ollama model is available locally via the /api/tags endpoint.

    Args:
        model_name (str): The name of the model to check (e.g., 'llama3', 'gemma:7b').
        ollama_url (str): The base URL of the Ollama API.

    Returns:
        Tuple[bool, Optional[str]]: (True if model exists, detailed error message or None).
                                     Returns (False, "Connection Error...") on connection issues.
    """
    api_tags_url = f"{ollama_url.rstrip('/')}/api/tags"
    target_model = model_name if ":" in model_name else f"{model_name}:latest"
    logging.debug(f"Checking for model '{target_model}' at {api_tags_url}")

    try:
        response = requests.get(
            api_tags_url, timeout=10
        )  # Short timeout for a quick check
        response.raise_for_status()
        data = response.json()
        models = data.get("models", [])

        for model in models:
            if model.get("name") == target_model:
                logging.info(f"Model '{target_model}' found locally.")
                return True, None  # Model found

        logging.info(f"Model '{target_model}' not found locally.")
        return False, None  # Model not found, but API reachable

    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection Error: Could not connect to Ollama API at {ollama_url}. Ensure Ollama is running. Details: {e}"
        logging.error(error_msg)
        return False, error_msg
    except requests.exceptions.Timeout:
        error_msg = f"Timeout error when checking for models at {ollama_url}."
        logging.error(error_msg)
        return False, error_msg
    except requests.exceptions.RequestException as e:
        error_msg = f"Error checking model availability: {e}"
        logging.error(error_msg)
        return False, error_msg
    except json.JSONDecodeError as e:
        error_msg = f"Failed to decode JSON response from {api_tags_url}: {e}"
        logging.error(error_msg)
        return False, error_msg


def download_ollama_model(model_name: str, ollama_url: str) -> bool:
    """
    Downloads the specified Ollama model using the /api/pull endpoint with streaming progress.

    Args:
        model_name (str): The name of the model to download.
        ollama_url (str): The base URL of the Ollama API.

    Returns:
        bool: True if download was successful or likely completed, False otherwise.
    """
    api_pull_url = f"{ollama_url.rstrip('/')}/api/pull"
    target_model = (
        model_name if ":" in model_name else f"{model_name}:latest"
    )  # Ensure tag for pull
    logging.info(f"Attempting to download model '{target_model}' from {ollama_url}...")
    logging.info("This may take some time depending on model size and network speed.")

    payload = {"name": target_model, "stream": True}  # Get progress updates

    try:
        # Use stream=True in requests to handle the streaming response efficiently
        with requests.post(
            api_pull_url, json=payload, stream=True, timeout=None
        ) as response:
            # Set timeout=None for the initial request, as the download duration is unknown.
            # The connection will stay open as long as the server streams data.

            response.raise_for_status()  # Check for initial errors like 404 Not Found

            last_status = ""
            download_complete = False
            for line in response.iter_lines():
                if line:
                    try:
                        decoded_line = line.decode("utf-8")
                        update = json.loads(decoded_line)
                        status = update.get("status", "")

                        # Print concise progress updates
                        if status != last_status:
                            print(
                                f"  Status: {status}"
                            )  # Use print for real-time feedback
                            last_status = status

                        if "total" in update and "completed" in update:
                            total = update["total"]
                            completed = update["completed"]
                            percent = (completed / total * 100) if total > 0 else 0
                            # Use carriage return to overwrite the progress line
                            print(
                                f"  Progress: {completed/1024/1024:.2f} MB / {total/1024/1024:.2f} MB ({percent:.1f}%)",
                                end="\r",
                            )

                        # Ollama's final status message for successful pull seems to be 'success'
                        if status == "success":
                            download_complete = True
                            print(
                                "\nDownload appears complete."
                            )  # Newline after progress bar
                            break  # Exit loop on success indication
                        # Handle potential errors reported in the stream
                        if "error" in update:
                            error_msg = update.get(
                                "error", "Unknown error during download."
                            )
                            logging.error(
                                f"\nError reported during download: {error_msg}"
                            )
                            print(
                                f"\nError during download: {error_msg}"
                            )  # Also print error
                            return False  # Download failed

                    except json.JSONDecodeError:
                        logging.warning(f"Could not decode JSON stream line: {line}")
                    except Exception as stream_err:
                        logging.error(f"Error processing stream line: {stream_err}")
                        print(f"\nError processing stream line: {stream_err}")
                        return False  # Indicate failure

            # After the loop, print a newline if progress was being printed
            if "total" in update:  # Check if progress indication was active
                print()  # Ensure the next log starts on a new line

            # Double check if the loop finished without explicit success/error
            if (
                not download_complete
                and last_status
                and "error" not in last_status.lower()
            ):
                logging.warning(
                    "Download stream finished without explicit 'success' status, but no error reported. Assuming completion."
                )
                download_complete = True  # Tentatively assume success if no error seen

            return download_complete

    except requests.exceptions.ConnectionError as e:
        logging.error(
            f"Connection Error: Could not connect to Ollama API at {ollama_url} to start download. Ensure Ollama is running. Details: {e}"
        )
        print(
            f"Connection Error: Could not connect to Ollama API at {ollama_url} to start download."
        )
        return False
    except requests.exceptions.Timeout:
        # This timeout would apply only to *initiating* the connection if timeout wasn't None.
        # With timeout=None, this is less likely unless the server never responds initially.
        logging.error(
            f"Timeout error when attempting to initiate download from {ollama_url}."
        )
        print("Timeout error trying to start the download.")
        return False
    except requests.exceptions.RequestException as e:
        # Handle errors like 404 (model not found on registry), 500, etc.
        logging.error(f"Error starting model download for '{target_model}': {e}")
        if hasattr(e, "response") and e.response is not None:
            logging.error(
                f"Response status: {e.response.status_code}, Body: {e.response.text[:200]}..."
            )  # Log part of body
            print(
                f"Error starting download ({e.response.status_code}). Check model name and Ollama server logs."
            )
        else:
            print(f"Error starting download: {e}")
        return False
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during model download: {e}", exc_info=True
        )
        print(f"An unexpected error occurred during download: {e}")
        return False


def ensure_model_exists(model_name: str, ollama_url: str) -> bool:
    """
    Checks if the model exists, and if not, attempts to download it.

    Args:
        model_name (str): The name of the model.
        ollama_url (str): The base URL of the Ollama API.

    Returns:
        bool: True if the model exists or was successfully downloaded, False otherwise.
    """
    model_exists, check_error = check_model_availability(model_name, ollama_url)

    if check_error:
        # Connection error or other issue checking availability - cannot proceed.
        logging.error(f"Failed to check model availability due to error: {check_error}")
        return False

    if not model_exists:
        logging.info(f"Model '{model_name}' not found locally. Attempting download...")
        if not download_ollama_model(model_name, ollama_url):
            logging.error(
                f"Failed to download model '{model_name}'. Please check the model name and ensure Ollama has network access."
            )
            return False  # Download failed
        else:
            logging.info(f"Model '{model_name}' downloaded successfully.")
            return True  # Download successful
    else:
        # Model already exists
        return True


# --- End Helper Functions ---


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments. Reads default values from environment
    variables if available, otherwise uses hardcoded defaults.
    Command-line arguments override environment variables.
    """
    parser = argparse.ArgumentParser(
        description="Process prompts from a CSV file using the Ollama API and save responses."
        " Defaults can be set via a .env file. Automatically downloads missing models."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        # required=True,
        default=os.getenv(
            "INPUT_CSV"
        ),  # Or allow via env if needed, but usually CLI is better
        help="Path to the input CSV file containing 'prompt_id' and 'prompt_text'.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        # required=True,
        default=os.getenv("OUTPUT_CSV"),  # Or allow via env if needed
        help="Path to save the output CSV file with responses.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL_NAME),
        help=f"Name of the Ollama model to use. Overrides OLLAMA_MODEL env var. (Default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=os.getenv("OLLAMA_URL", OllamaClient.DEFAULT_OLLAMA_URL),
        help=f"Base URL of the Ollama API. Overrides OLLAMA_URL env var. (Default: {OllamaClient.DEFAULT_OLLAMA_URL})",
    )
    # ... (rest of the arguments remain the same)
    parser.add_argument(
        "--temperature",
        type=lambda v: float(v) if v is not None else None,
        default=os.getenv(
            "OLLAMA_TEMPERATURE"
        ),  # Default to None if not set, Client handles None
        help="LLM temperature. Overrides OLLAMA_TEMPERATURE env var.",
    )
    parser.add_argument(
        "--top-k",
        type=lambda v: int(v) if v is not None else None,
        default=os.getenv("OLLAMA_TOP_K"),
        help="LLM top_k parameter. Overrides OLLAMA_TOP_K env var.",
    )
    parser.add_argument(
        "--top-p",
        type=lambda v: float(v) if v is not None else None,
        default=os.getenv("OLLAMA_TOP_P"),
        help="LLM top_p parameter. Overrides OLLAMA_TOP_P env var.",
    )
    parser.add_argument(
        "--context-window",
        type=lambda v: int(v) if v is not None else None,
        default=os.getenv("OLLAMA_CONTEXT_WINDOW"),
        help="LLM context window size (num_ctx). Overrides OLLAMA_CONTEXT_WINDOW env var.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=os.getenv("OLLAMA_TIMEOUT", OllamaClient.DEFAULT_TIMEOUT),
        help=f"API request timeout in seconds (for /api/generate). Overrides OLLAMA_TIMEOUT env var. (Default: {OllamaClient.DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=os.getenv("PROCESS_DELAY", 0.5),
        help="Delay in seconds between API calls. Overrides PROCESS_DELAY env var. (Default: 0.5)",
    )

    args = parser.parse_args()

    # --- Argument Type Conversion/Validation ---
    # Helper to safely convert env vars (strings) or CLI args to target types
    def safe_convert(value, target_type, default=None):
        if value is None:
            return default
        try:
            return target_type(value)
        except (ValueError, TypeError):
            logging.warning(
                f"Could not convert value '{value}' to {target_type}. Using default: {default}."
            )
            return default

    # Apply conversions, using the value already parsed by argparse as the input 'value'
    # This handles both CLI args (already typed) and env var defaults (might be strings)
    args.temperature = safe_convert(
        args.temperature, float
    )  # Default None if conversion fails or not set
    args.top_k = safe_convert(args.top_k, int)
    args.top_p = safe_convert(args.top_p, float)
    args.context_window = safe_convert(args.context_window, int)
    # timeout and delay use getenv defaults that are already typed, should be ok
    args.timeout = int(args.timeout)  # Ensure int
    args.delay = float(args.delay)  # Ensure float

    # Final check: Ensure a model is specified
    if not args.model:
        parser.error(
            "The 'model' argument is required, either via --model, OLLAMA_MODEL env var, or the default."
        )

    # Convert potential string numbers from env vars if they were used as defaults
    # (Argparse handles CLI input typing automatically)
    if isinstance(args.temperature, str):
        args.temperature = safe_convert(args.temperature, float)
    if isinstance(args.top_k, str):
        args.top_k = safe_convert(args.top_k, int)
    if isinstance(args.top_p, str):
        args.top_p = safe_convert(args.top_p, float)
    if isinstance(args.context_window, str):
        args.context_window = safe_convert(args.context_window, int)
    if isinstance(args.timeout, str):
        args.timeout = safe_convert(args.timeout, int, OllamaClient.DEFAULT_TIMEOUT)
    if isinstance(args.delay, str):
        args.delay = safe_convert(args.delay, float, 0.5)

    return args


def read_prompts(input_csv_path: Path) -> List[Dict[str, str]]:
    """Reads prompts from the input CSV file."""
    prompts = []
    if not input_csv_path:
        logging.error("Input CSV path is not specified.")
        raise ValueError("Input CSV path cannot be empty.")
    if not input_csv_path.is_file():
        logging.error(f"Input CSV file not found: {input_csv_path}")
        raise FileNotFoundError(f"Input CSV file not found: {input_csv_path}")

    try:
        with open(input_csv_path, mode="r", encoding="utf-8", newline="") as infile:
            reader = csv.DictReader(infile)
            if (
                not reader.fieldnames
                or "prompt_id" not in reader.fieldnames
                or "prompt_text" not in reader.fieldnames
            ):
                raise ValueError(
                    "Input CSV must contain 'prompt_id' and 'prompt_text' columns."
                )
            for row_num, row in enumerate(
                reader, start=2
            ):  # Start at 2 for header row + 1-based index
                prompt_id = row.get("prompt_id", "").strip()
                prompt_text = row.get("prompt_text", "").strip()

                if prompt_id and prompt_text:
                    prompts.append({"prompt_id": prompt_id, "prompt_text": prompt_text})
                else:
                    logging.warning(
                        f"Skipping row {row_num} due to missing prompt_id or prompt_text: {row}"
                    )
    except Exception as e:
        logging.error(f"Error reading input CSV file {input_csv_path}: {e}")
        raise
    return prompts


def process_prompts(
    prompts: List[Dict[str, str]],
    client: OllamaClient,
    output_csv_path: Path,
    expected_categories: List[str],
    delay: float,
) -> None:
    """Processes prompts using the Ollama client and writes results to the output CSV."""
    output_fieldnames = [
        "prompt_id",
        "prompt_text",
        "raw_llm_response",
        "cleaned_response",
        "fail_status",  # True if API call failed OR validation failed
    ]
    if not output_csv_path:
        logging.error("Output CSV path is not specified.")
        raise ValueError("Output CSV path cannot be empty.")

    # Ensure output directory exists
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_csv_path, mode="w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
            writer.writeheader()

            total_prompts = len(prompts)
            logging.info(f"Starting processing of {total_prompts} prompts...")

            for i, prompt_data in enumerate(prompts):
                prompt_id = prompt_data["prompt_id"]
                prompt_text = prompt_data["prompt_text"]
                logging.info(
                    f"Processing prompt {i+1}/{total_prompts} (ID: {prompt_id})..."
                )

                raw_response, api_success = client.generate(prompt_text)

                cleaned_response = ""
                validation_passed = False
                fail_status = True  # Assume failure initially

                if api_success and raw_response is not None:
                    logging.debug(
                        f"Prompt ID {prompt_id}: Received raw response."
                    )  # Debug for less noise
                    cleaned_response, validation_passed = validate_and_clean_response(
                        raw_response, expected_categories
                    )
                    if validation_passed:
                        logging.info(
                            f"Prompt ID {prompt_id}: Validation successful. Category: '{cleaned_response}'"
                        )
                        fail_status = (
                            False  # Success! API call worked AND validation passed
                        )
                    else:
                        logging.warning(
                            f"Prompt ID {prompt_id}: Validation FAILED. Raw: '{raw_response[:100]}...', Cleaned: '{cleaned_response}'"
                        )  # Log snippet
                        fail_status = True  # API call worked, but validation failed
                else:
                    logging.error(f"Prompt ID {prompt_id}: API call failed.")
                    # raw_response might be None here from the client error handling
                    raw_response = (
                        raw_response if raw_response is not None else "API_CALL_FAILED"
                    )  # Ensure something is written
                    fail_status = True  # API call itself failed

                # Write result row
                writer.writerow(
                    {
                        "prompt_id": prompt_id,
                        "prompt_text": prompt_text,
                        "raw_llm_response": raw_response,
                        "cleaned_response": cleaned_response,
                        "fail_status": fail_status,
                    }
                )

                # Optional delay between requests
                if delay > 0 and i < total_prompts - 1:
                    logging.debug(f"Waiting for {delay} seconds before next request.")
                    time.sleep(delay)

            logging.info(f"Finished processing. Results saved to {output_csv_path}")

    except IOError as e:
        logging.error(f"Error writing to output CSV file {output_csv_path}: {e}")
        raise
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during prompt processing: {e}", exc_info=True
        )
        raise


def main():
    """Main function to run the prompt processing pipeline."""
    args = parse_arguments()

    try:
        # 0. Ensure Ollama is reachable and the model exists/is downloaded
        logging.info("Checking Ollama connection and model availability...")
        if not ensure_model_exists(args.model, args.ollama_url):
            # Error message already logged by ensure_model_exists or its helpers
            print(f"Exiting due to issues with Ollama or model '{args.model}'.")
            sys.exit(1)  # Exit if model check/download fails
        logging.info("Ollama connection ok and model is available.")

        # 1. Initialize Ollama Client
        client = OllamaClient(
            model=args.model,
            ollama_url=args.ollama_url,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            context_window=args.context_window,
            timeout=args.timeout,
        )

        # 2. Read Prompts
        prompts = read_prompts(args.input_csv)
        if not prompts:
            logging.warning("No valid prompts found in the input file. Exiting.")
            return  # Exit cleanly if no prompts

        # 3. Process Prompts and Write Output
        process_prompts(
            prompts=prompts,
            client=client,
            output_csv_path=args.output_csv,
            expected_categories=EXPECTED_CATEGORIES,
            delay=args.delay,
        )

    except (ValueError, FileNotFoundError) as ve:
        logging.error(f"Configuration, File, or Input Data Error: {ve}")
        print(f"Error: {ve}")
        sys.exit(1)  # Exit on file/config errors too
    except Exception as e:
        logging.error(
            f"A critical error occurred in the main execution flow: {e}", exc_info=True
        )
        print(f"An unexpected critical error occurred: {e}")
        sys.exit(1)  # Exit on other critical errors


if __name__ == "__main__":
    main()
