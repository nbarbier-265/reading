#!/bin/bash
brew install uv
uv venv amira-project-env --python=3.11
source amira-project-env/bin/activate
uv pip install -r requirements.txt

ln -s $(pwd)/config app/config

python src/skill_match.py 
python src/intervention_points.py

export PYTHONPATH="$(pwd):$PYTHONPATH"
streamlit run app/reader.py