# Story Skill Matcher

Documentation provided inside the streamlit app.

## Features

## Setup

1. Run the setup script (dependencies: MacOS and homebrew), which setups up python, your virtual env, and installs dependencies, processes the data and starts the app:
  ```
  sh setup_reprocess_examples.sh
  ```

2. Set up your OpenAI API key and associated model in the `.env` file:
  (the model must support chat completions)
   ```
   OPENAI_API_KEY=your_api_key_here
   OPEN_AI_MODEL=your_model_here
   ```

3. Alternatively, use cached data with:
  ```
  sh setup.sh
  ```