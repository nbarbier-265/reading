# Story Skill Matcher

This project analyzes stories and matches them to relevant skills using LLM-based text analysis. The system assigns scores to indicate the relevance of each story to various skill categories.

## Features

## Setup

1. Run the setup script (assumes MacOS and homebrew), which setups up python, your virtual env, and installs dependencies, and starts the app:
  ```
  sh setup.sh
  ```

2. Set up your OpenAI API key and associated model in the `.env` file:
  (the model must support chat completions)
   ```
   OPENAI_API_KEY=your_api_key_here
   OPEN_AI_MODEL=your_model_here
   ```