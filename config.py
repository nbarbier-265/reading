import os

PATH_TO_PROJECT = os.path.dirname(os.path.abspath(__file__))

# Set the project path as an environment variable
os.environ["PATH_TO_PROJECT"] = PATH_TO_PROJECT

PATH_TO_DATA = os.path.join(PATH_TO_PROJECT, "data")
PATH_TO_SOURCE_DATA = os.path.join(PATH_TO_DATA, "source_data")
PATH_TO_PROCESSED_DATA = os.path.join(PATH_TO_DATA, "processed_data")
