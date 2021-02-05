import os


def ensure_folder(path):
    # create folder if does not exist
    if os.path.isdir(path):
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder)
