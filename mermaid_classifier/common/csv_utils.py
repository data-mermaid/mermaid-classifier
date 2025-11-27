import abc
import typing

import pandas as pd


def csv_to_dataframe(csv_file: typing.TextIO):
    """
    CSV file to pandas dataframe. Use this for non-mathematical data
    such as strings or integers that just represent IDs.

    Throws pd.errors.EmptyDataError if there's no CSV data.
    """
    return pd.read_csv(
        csv_file,
        # Don't convert blank cells to NaN. It seems easier to
        # just check if row.get('...') is truthy, than to
        # check if it's NaN, when we're not dealing with mathematical
        # data in the first place.
        keep_default_na=False,
    )


class CsvSpec(abc.ABC):
    """
    A CSV file given as input to specify something, such as sources
    to accept or benthic attributes to roll up to.

    Subclasses should define:
    - target_column_candidates: The first column name in this list
      which is present in the CSV becomes the 'target column'. Every
      row in the CSV must have a value for the target column.
    - per_item_init_action(): This base class's __init__() will call
      this method for every row in the CSV.
    """

    target_column_candidates: list[str]

    def __init__(self, csv_file: typing.TextIO):

        try:
            self.csv_dataframe = csv_to_dataframe(csv_file)
        except pd.errors.EmptyDataError:
            # It just errors if there's no CSV data, so we manually
            # create an empty dataframe in this case.
            # We also short-circuit to prevent any other odd cases.
            self.csv_dataframe = pd.DataFrame()
            return

        csv_filename = getattr(csv_file, 'name', "<File-like obj>")

        target_column = None
        candidates_str = ", ".join(self.target_column_candidates)
        for column_name in self.target_column_candidates:
            if column_name in self.csv_dataframe.columns:
                target_column = column_name
        if target_column is None:
            raise ValueError(
                f"{csv_filename}: must have at least"
                f" one of these columns: {candidates_str}")

        for row_i, (_, row) in enumerate(self.csv_dataframe.iterrows()):
            target_value = row.get(target_column)
            if not target_value:
                raise ValueError(
                    f"{csv_filename}:"
                    f"{target_column} not found in row {row_i + 1}")

            self.per_item_init_action(target_column, target_value)

    def per_item_init_action(self, target_column, target_value):
        raise NotImplementedError
