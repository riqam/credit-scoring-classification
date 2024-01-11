import yaml
import joblib
from datetime import datetime

CONFIG_DIR = 'config/config.yaml'

# Function to load config files
def config_load():
    try:
        with open(CONFIG_DIR, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as error:
        raise RuntimeError('Parameters file not found in path.')
    
    return config

# Function to load pickle files
def pickle_load(file_path):
    return joblib.load(file_path)

# Function to dump data into pickle
def pickle_dump(data, file_path):
    joblib.dump(data, file_path)

# Function that return the current time
def time_stamp():
    return datetime.now()