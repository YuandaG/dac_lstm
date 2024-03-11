import os

def get_single_filename(folder_path):
    files = os.listdir(folder_path)
    if len(files) == 1:
        return files[0]
    else:
        print("There is not exactly one file in the folder.")
        return None