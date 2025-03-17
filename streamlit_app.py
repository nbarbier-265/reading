"""
Main entry point for the Streamlit application.

This file serves as the entry point for Streamlit Cloud deployments.
"""

import os
import sys

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up environment for the application
os.environ["PATH_TO_PROJECT"] = os.path.dirname(os.path.abspath(__file__))

# Import the main function from the app
from app.main import main

# Run the application
if __name__ == "__main__":
    main() 