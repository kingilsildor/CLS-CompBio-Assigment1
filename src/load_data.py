import os

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
        Load data from a CSV file.

        Params
        -------
        - file_path (str): The path to the CSV file.
    str
        Returns
        -------
        - pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    try:
        data = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file {file_path} is empty.")
    except pd.errors.ParserError:
        raise ValueError(f"The file {file_path} could not be parsed.")

    return data
