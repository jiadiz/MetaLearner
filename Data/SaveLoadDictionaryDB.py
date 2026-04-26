import pickle
from typing import Any
import os
from pathlib import Path


DEFAULT_DATAFILES_DIR = Path(__file__).resolve().parents[1] / "Datafiles"


def _resolve_database_directory(directory: str | Path | None) -> Path:
    if directory is None:
        return DEFAULT_DATAFILES_DIR
    return Path(directory)

def load_dictionary_database(
    directory: str | Path | None = None,
    filename: str = "mean_reversion_base_feature_database.pkl",
) -> Any:
    """
    Load dictionary database from `directory/filename` if it exists.
    If not found, create an empty dict file and return {}.
    """
    database_dir = _resolve_database_directory(directory)
    os.makedirs(database_dir, exist_ok=True)
    file_path = database_dir / filename
    if file_path.is_file():
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
    directory: str | Path | None = None,
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
    database_dir = _resolve_database_directory(directory)
    os.makedirs(database_dir, exist_ok=True)
    file_path = database_dir / filename
    with open(file_path, "wb") as f:
        pickle.dump(database, f, protocol=pickle.HIGHEST_PROTOCOL)
    return str(file_path)
