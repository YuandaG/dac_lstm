import os

def check_path(path):
    if not os.path.exists(path):
    # Create the directory if it doesn't exist
        os.makedirs(path)