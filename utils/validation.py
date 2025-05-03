# utils/validation.py

import re
from typing import List, Tuple, Optional


def validate_and_clean_response(
    response_text: Optional[str], expected_categories: List[str]
) -> Tuple[str, bool]:
    """
    Validates if the response text contains one of the expected categories
    and cleans it up.

    Args:
        response_text (Optional[str]): The raw response text from the LLM.
        expected_categories (List[str]): A list of valid category strings.

    Returns:
        Tuple[str, bool]: A tuple containing:
            - The cleaned response (the found category) or the original cleaned text if no category found.
            - A boolean indicating if a valid category was found (True) or not (False).
    """
    if not response_text:
        return "", False

    # Basic cleaning: remove leading/trailing whitespace and quotes
    cleaned_text = response_text.strip().strip('"').strip("'").strip()

    # Case-insensitive matching
    cleaned_text_lower = cleaned_text.lower()
    expected_categories_lower = {
        cat.lower(): cat for cat in expected_categories
    }  # Map lower to original case

    found_category = ""
    is_valid = False

    # Try to find an exact match first (most reliable if LLM follows instructions)
    if cleaned_text_lower in expected_categories_lower:
        found_category = expected_categories_lower[cleaned_text_lower]
        is_valid = True
    else:
        # If no exact match, check if any category is a substring
        # This is less precise but handles cases where the LLM adds extra text.
        # Prioritize longer matches if multiple categories are substrings (e.g., "Positive Experience" vs "Positive")
        # This simple check might need refinement depending on LLM verbosity.
        possible_matches = []
        for cat_lower, cat_original in expected_categories_lower.items():
            # Use regex word boundary to avoid partial matches like 'bug' in 'debug'
            # Or simply check if the whole category name is present
            # A simpler check: if cat_lower in cleaned_text_lower:
            if re.search(
                r"\b" + re.escape(cat_lower) + r"\b", cleaned_text_lower, re.IGNORECASE
            ):
                possible_matches.append(cat_original)

        if len(possible_matches) == 1:
            found_category = possible_matches[0]
            is_valid = True
        elif len(possible_matches) > 1:
            # Handle ambiguity - perhaps just take the first one found, or mark as invalid?
            # For now, let's consider it valid but maybe log a warning or choose longest match.
            # Choosing longest match as a simple heuristic:
            found_category = max(possible_matches, key=len)
            is_valid = True
            # Or mark as invalid due to ambiguity:
            # found_category = cleaned_text # Keep original if ambiguous
            # is_valid = False
        else:
            # No category found
            found_category = cleaned_text  # Return the cleaned original text
            is_valid = False

    # If a valid category was found, return only that category name.
    # Otherwise, return the cleaned-up original response.
    final_response = found_category if is_valid else cleaned_text
    return final_response, is_valid


# Example Usage
if __name__ == "__main__":
    cats = [
        "Positive Experience",
        "Negative Experience",
        "Neutral Feedback",
        "Feature Request",
        "Bug Report",
        "Other",
    ]

    test_cases = [
        "Positive Experience",
        '"Negative Experience"',
        "  Neutral Feedback  ",
        "Feature Request.",
        "The user reported a Bug Report.",
        "bug report",
        "This is Other.",
        "I think it's positive experience",  # Should find "Positive Experience"
        "No specific category mentioned.",
        None,
        "",
        "Positive",  # Ambiguous if "Positive Experience" is expected - current logic might miss this unless 'Positive' is a category
        "This is both Positive Experience and Negative Experience",  # Ambiguous
    ]

    print("--- Validation Tests ---")
    for text in test_cases:
        cleaned, valid = validate_and_clean_response(text, cats)
        print(f"Original: '{text}' -> Cleaned: '{cleaned}', Valid: {valid}")
    print("----------------------")
