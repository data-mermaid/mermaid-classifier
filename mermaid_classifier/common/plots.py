from collections import defaultdict
import colorsys
from dataclasses import dataclass
from typing import Any

from matplotlib import patheffects
from matplotlib.lines import Line2D
import numpy as np


@dataclass
class PointMarker:
    """
    row
    col
    Pixel position of the point marker on the image.

    color
    Color in any format matplotlib accepts, such as an
    RGBA 4-tuple of floats ranging from 0.0 to 1.0.

    shape
    https://matplotlib.org/stable/api/markers_api.html

    text
    Text to write next to the point marker using Axes.annotate().
    """
    row: int
    col: int
    color: Any
    shape: Any
    text: str = None


EDGE_COLOR = 'black'
EDGE_WIDTH = 0.5


def plot_point_markers(ax: 'Axes', markers: list[PointMarker]):

    marker_groups = defaultdict(list)
    marker_size = 100.0
    font_size = 11
    text_offset = 4.0

    for marker in markers:
        if (
            isinstance(marker.color, list)
            or isinstance(marker.color, np.ndarray)
        ):
            hashable_color = tuple(marker.color)
        else:
            hashable_color = marker.color

        # Group the markers according to their common properties other
        # than row, col, text.
        group_identifying_dict = dict(
            color=hashable_color,
            shape=marker.shape,
        )
        group_hash = tuple(
            (k, v) for k, v in group_identifying_dict.items()
        )
        marker_groups[group_hash].append(marker)

    for group_hash, markers in marker_groups.items():

        xs = [marker.col for marker in markers]
        ys = [marker.row for marker in markers]
        group_dict = dict(group_hash)

        # Marker
        ax.scatter(
            xs,
            ys,
            marker=group_dict['shape'],
            color=group_dict['color'],
            # Bit of transparency helps to make the marker not stand
            # out too much.
            alpha=0.8,
            s=marker_size,
            # Marker edges make the markers easier to see when the
            # marker color is similar to the image color.
            edgecolors=EDGE_COLOR, linewidths=EDGE_WIDTH,
        )

        # Text
        for marker in markers:
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html
            ax.annotate(
                marker.text,
                (marker.col, marker.row),
                color=group_dict['color'],
                alpha=0.8,
                fontsize=font_size,
                # https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text.set_fontweight
                fontweight='semibold',
                # textcoords in offset units allow the text to be a
                # consistent distance from the point marker, regardless
                # of image scale or zoom level.
                # This makes sense because the marker size doesn't
                # scale along with the zoom level.
                # TODO: Might be better to position at bottom left of marker,
                #  rather than top right, so it doesn't risk overlapping with
                #  the legend or the plot title. But just passing negative
                #  values here doesn't do it; probably also need to adjust the
                #  anchor.
                xytext=(text_offset, text_offset),
                textcoords='offset points',
                # Border/outline around the text for better visibility when
                # the marker color is similar to the image color.
                # https://github.com/has2k1/plotnine/discussions/898
                path_effects=[
                    patheffects.Stroke(linewidth=1, foreground=EDGE_COLOR),
                    patheffects.Normal(),
                ],
            )


@dataclass
class LegendSpecElement:
    # Marker color in legend, in any color format matplotlib accepts.
    color: Any
    # Marker shape in legend, in any shape format matplotlib accepts.
    shape: Any
    # Text next to the marker.
    label: str


def plot_legend(
    ax: 'Axes',
    spec: list[LegendSpecElement],
    title: str = None,
    position_kwargs: dict = None
) -> 'matplotlib.legend.Legend':

    legend_artists = [
        Line2D(
            # We don't care about this numeric data; this is
            # just a placeholder for the legend.
            [0],[0],
            marker=spec_element.shape, linestyle='None', markersize=8,
            markerfacecolor=spec_element.color,
            markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
            label=spec_element.label,
        )
        for spec_element in spec
    ]

    if position_kwargs is None:
        # Put the legend outside the plot, to the upper right.
        position_kwargs = dict(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    # This call already adds the legend to the axes, but the legend
    # handle may be needed for things like adding multiple legends, so
    # we return it.
    return ax.legend(
        handles=legend_artists,
        # Spacing between the legend items
        labelspacing=1,
        title=title,
        **position_kwargs
    )


def adjust_lightness(rgba, l_factor):
    """
    Adjust lightness of a matplotlib rgba color.
    https://stackoverflow.com/a/60562502
    """
    r, g, b, a = rgba
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    new_rgb = colorsys.hls_to_rgb(h, min(1.0, l * l_factor), s=s)
    return [*new_rgb, a]


def adjust_saturation(rgba, s_factor):
    r, g, b, a = rgba
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    new_rgb = colorsys.hls_to_rgb(h, l, s=min(1.0, s * s_factor))
    return [*new_rgb, a]
