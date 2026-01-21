import abc
import dataclasses
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


@dataclasses.dataclass
class ColumnSpec:
    name: str | list[str]
    allow_blank: bool = True


class CsvSpec(abc.ABC):
    """
    A CSV file given as input to specify something, such as sources
    to accept or benthic attributes to roll up to.

    Subclasses should define:
    - required_columns: Each list item here represents one required column. Each item can be A) the name of the required column, or B) a list of accepted names for the required column, and the first such name
      which is present in the CSV is the name that's used.
    - per_item_init_action(): This base class's __init__() will call
      this method for every row in the CSV.
    """
    column_specs: list[ColumnSpec]

    def __init__(self, csv_file: typing.TextIO):

        self.csv_text = csv_file.read()
        # Set up for re-reading with pandas.
        csv_file.seek(0)

        try:
            self.csv_dataframe = csv_to_dataframe(csv_file)
        except pd.errors.EmptyDataError:
            # It just errors if there's no CSV data, so we manually
            # create an empty dataframe in this case.
            # We also short-circuit to prevent any other odd cases.
            self.csv_dataframe = pd.DataFrame()
            return

        csv_filename = getattr(csv_file, 'name', "<File-like obj>")

        column_names = []

        for column_spec in self.column_specs:
            if isinstance(column_spec.name, str):
                if column_spec.name in self.csv_dataframe.columns:
                    column_name = column_spec.name
                else:
                    raise ValueError(
                        f"{csv_filename}: must have the column"
                        f" {column_spec.name}")
            else:
                # List of str
                column_name = None
                for candidate_name in column_spec.name:
                    if candidate_name in self.csv_dataframe.columns:
                        column_name = candidate_name
                        break
                if column_name is None:
                    candidates_str = ", ".join(column_spec.name)
                    raise ValueError(
                        f"{csv_filename}: must have at least"
                        f" one of these columns: {candidates_str}")
            column_names.append(column_name)

        for row_i, (_, row) in enumerate(self.csv_dataframe.iterrows()):
            values = dict()
            for spec, name in zip(self.column_specs, column_names):
                # Ensure absent values are None, not ''.
                value = row.get(name) or None

                if not spec.allow_blank and not value:
                    raise ValueError(
                        f"{csv_filename}:"
                        f"{name} not found in row {row_i + 1}")

                values[name] = value

            self.per_row_init_action(values)

    def per_row_init_action(self, row: dict):
        raise NotImplementedError
