
import pickle
from typing import Any
import os

def load_dictionary_database(
    directory: str = r"MetaLearner\Datafiles",
    filename: str = "mean_reversion_base_feature_database.pkl",
) -> Any:
    """
    Load dictionary database from `directory/filename` if it exists.
    If not found, create an empty dict file and return {}.
    """
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
        print(f"Database found: {file_path}")
        with open(file_path, "rb") as f:
            return pickle.load(f)
    print(f"Database not found. Creating new empty database: {file_path}")
    empty_db = {}
    with open(file_path, "wb") as f:
        pickle.dump(empty_db, f, protocol=pickle.HIGHEST_PROTOCOL)
    return empty_db


def save_dictionary_database(
    database: Any,
    directory: str = r"MetaLearner\Datafiles",
    filename: str = "mean_reversion_base_feature_database.pkl",
) -> str:
    """
    Save `database` to a pickle file at `directory/filename`.
    Creates the directory if it does not exist.

    Returns
    -------
    str
        Full path of the saved pickle file.
    """
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, "wb") as f:
        pickle.dump(database, f, protocol=pickle.HIGHEST_PROTOCOL)
    return file_path
