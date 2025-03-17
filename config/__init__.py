"""
Configuration module for the Amira application.

This module provides configuration variables used across the application.
"""

import os
from pathlib import Path

# Path configuration
PATH_TO_PROJECT = os.environ.get("PATH_TO_PROJECT", str(Path.cwd()))
PATH_TO_SOURCE_DATA = os.path.join(PATH_TO_PROJECT, "data", "source_data")
PATH_TO_PROCESSED_DATA = os.path.join(PATH_TO_PROJECT, "data", "processed_data")

# Add prompt version configuration
PATH_TO_PROMPTS = os.path.join(PATH_TO_PROJECT, "config", "prompts")
DEFAULT_PROMPT_VERSION = "v1"


# Create a function to get the current prompt version path
def get_prompt_version_path(version=None):
    """
    Get the path to the prompt version directory.

    Args:
        version (str, optional): The prompt version. If None, uses DEFAULT_PROMPT_VERSION.

    Returns:
        str: The path to the prompt version directory
    """
    version = version or DEFAULT_PROMPT_VERSION
    return os.path.join(PATH_TO_PROMPTS, version)


# Create a function to get the path for processed data by prompt version
def get_processed_data_path(prompt_version=None):
    """
    Get the path to store processed data based on prompt version.

    Args:
        prompt_version (str, optional): The prompt version. If None, uses DEFAULT_PROMPT_VERSION.

    Returns:
        str: The path to the processed data for this prompt version
    """
    version = prompt_version or DEFAULT_PROMPT_VERSION
    return os.path.join(PATH_TO_PROCESSED_DATA, f"prompt_{version}")
