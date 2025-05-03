# ollama_client/client.py

import requests
import json
import logging
from typing import Optional, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OllamaClient:
    """
    A client for interacting with the Ollama API's /api/generate endpoint.

    Allows sending prompts to a specified model and retrieving responses,
    with configurable LLM parameters.
    """

    DEFAULT_OLLAMA_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 60  # seconds

    def __init__(
        self,
        model: str,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        temperature: Optional[float] = 0.7,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        context_window: Optional[int] = 2048,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """
        Initializes the OllamaClient.

        Args:
            model (str): The name of the Ollama model to use (e.g., 'llama3').
            ollama_url (str): The base URL of the Ollama API endpoint.
            temperature (Optional[float]): Controls randomness. Lower is more deterministic.
            top_k (Optional[int]): Reduces the probability mass considered nucleus sampling.
            top_p (Optional[float]): Limits sampling to the top K most probable tokens.
            context_window (Optional[int]): The size of the context window in tokens.
            timeout (int): Request timeout in seconds.
        """
        if not model:
            raise ValueError("Model name cannot be empty.")

        self.ollama_url = ollama_url.rstrip("/")
        self.api_endpoint = f"{self.ollama_url}/api/generate"
        self.model = model
        self.timeout = timeout

        # Build the options dictionary, including only non-None parameters
        self.options = {}
        if temperature is not None:
            self.options["temperature"] = temperature
        if top_k is not None:
            self.options["top_k"] = top_k
        if top_p is not None:
            self.options["top_p"] = top_p
        if context_window is not None:
            # Note: Ollama typically uses 'num_ctx' for context window size
            self.options["num_ctx"] = context_window
            # Add other common Ollama options if needed, e.g., seed, mirostat, etc.

        logging.info(
            f"OllamaClient initialized for model '{self.model}' at {self.ollama_url}"
        )
        logging.info(f"Using options: {self.options}")

    def generate(self, prompt: str) -> Tuple[Optional[str], bool]:
        """
        Sends a prompt to the Ollama API and retrieves the generated response.

        Args:
            prompt (str): The input prompt text.

        Returns:
            Tuple[Optional[str], bool]: A tuple containing:
                - The generated response text (str) or None if an error occurred.
                - A boolean success flag (True if successful, False otherwise).
        """
        if not prompt:
            logging.warning("Received empty prompt.")
            return None, False

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # Get the full response at once
        }
        if self.options:
            payload["options"] = self.options

        logging.debug(f"Sending payload to {self.api_endpoint}: {payload}")

        try:
            response = requests.post(
                self.api_endpoint,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            response_data = response.json()
            generated_text = response_data.get("response")

            if generated_text:
                logging.debug(
                    f"Received response: {generated_text[:100]}..."
                )  # Log snippet
                return generated_text.strip(), True
            else:
                logging.error(
                    f"API response did not contain 'response' field. Full response: {response_data}"
                )
                return None, False

        except requests.exceptions.ConnectionError as e:
            logging.error(
                f"Connection error: Could not connect to Ollama API at {self.ollama_url}. Ensure Ollama is running. Details: {e}"
            )
            return None, False
        except requests.exceptions.Timeout as e:
            logging.error(
                f"Request timed out after {self.timeout} seconds. Details: {e}"
            )
            return None, False
        except requests.exceptions.RequestException as e:
            logging.error(f"An error occurred during the API request: {e}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    # Try to get more specific error details from Ollama response
                    error_details = e.response.json()
                    logging.error(f"Ollama API error details: {error_details}")
                except json.JSONDecodeError:
                    logging.error(
                        f"Could not decode error response body: {e.response.text}"
                    )
            return None, False
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON response from API: {e}")
            return None, False
        except Exception as e:
            logging.error(
                f"An unexpected error occurred in generate method: {e}", exc_info=True
            )
            return None, False


# Example Usage (can be run standalone for testing)
if __name__ == "__main__":
    # Ensure Ollama is running with a model like 'llama3' or 'mistral'
    try:
        # Test with a valid model available in your Ollama instance
        test_model = "llama3"  # CHANGE THIS if needed
        client = OllamaClient(model=test_model, temperature=0.5)

        test_prompt = "Why is the sky blue?"
        print(f"Sending test prompt: '{test_prompt}' to model '{test_model}'...")

        response_text, success = client.generate(test_prompt)

        if success:
            print("\n--- Test Response ---")
            print(response_text)
            print("--------------------")
        else:
            print("\n--- Test Failed ---")
            print("Could not get response from Ollama API.")
            print(
                "Please ensure Ollama is running and the model",
                test_model,
                "is available.",
            )
            print(f"Check the API URL: {client.ollama_url}")

        # Test connection error (use a bad URL)
        print("\n--- Testing Connection Error ---")
        bad_client = OllamaClient(model=test_model, ollama_url="http://localhost:11435")
        _, success = bad_client.generate(test_prompt)
        if not success:
            print("Connection error test successful (failed as expected).")
        else:
            print("Connection error test failed (unexpected success).")

    except ValueError as ve:
        print(f"Configuration error: {ve}")
    except Exception as ex:
        print(f"An unexpected error occurred during testing: {ex}")
