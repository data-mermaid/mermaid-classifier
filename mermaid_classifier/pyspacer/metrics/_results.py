"""Structured output types returned by metric functions."""

import dataclasses
import typing

import pandas as pd
from matplotlib.figure import Figure


@dataclasses.dataclass
class ScalarMetric:
    """A single named numeric metric."""

    name: str
    value: float


@dataclasses.dataclass
class FigureResult:
    """A matplotlib figure to log as an artifact."""

    fig: Figure
    artifact_path: str


@dataclasses.dataclass
class DataFrameResult:
    """A DataFrame to log as a CSV artifact."""

    df: pd.DataFrame
    artifact_path: str


@dataclasses.dataclass
class DictResult:
    """A dict to log as a YAML/JSON artifact."""

    data: dict[str, typing.Any]
    artifact_path: str


@dataclasses.dataclass
class MetricGroupResult:
    """Collection of results from a single metric group."""

    scalars: list[ScalarMetric] = dataclasses.field(default_factory=list)
    figures: list[FigureResult] = dataclasses.field(default_factory=list)
    dataframes: list[DataFrameResult] = dataclasses.field(default_factory=list)
    dicts: list[DictResult] = dataclasses.field(default_factory=list)
