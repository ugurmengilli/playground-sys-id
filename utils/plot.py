import math

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots


def get_figure(
        y_data: [ArrayLike, list[ArrayLike]],
        label: [str, list[str]],
        y_title: str,
        x_data: ArrayLike = None,
        x_title: str = None,
        title: str = None
) -> Figure:
    """
    Plot a (list of) ``y_data`` with a (list of) ``label`` in a single
    plot. If ``x_data`` is not provided, it will be generated as a
    range of the maximum length of the data. Otherwise, the provided
    ``x_data`` will be used as the x-axis of all the channels.

    Args:
        y_data: (list of) list(s) of floats, data to plot.
        label: (list of) string(s), labels for each channel.
        y_title: title for the y-axis.
        x_data: data for the x-axis.
        x_title: title for the x-axis.
        title: title for the plot.

    Returns:
        fig: the plot.
    """
    fig = go.Figure()

    # Ensure that y_data and label are lists to pass them to the loop.
    y_data = y_data if isinstance(y_data, list) else [y_data]
    label = label if isinstance(label, list) else [label]

    # If x_data is not provided, generate it as a range of the maximum
    # length of the data.
    x_data = x_data if x_data is not None\
        else list(range(max([len(y) for y in y_data])))

    # Plot each channel.
    for channel, name in zip(y_data, label):
        fig.add_trace(go.Scatter(
            x=x_data, y=channel, mode='lines', name=name
        ))

    fig.update_layout(
        title=title if title else None,
        xaxis=dict(title=x_title),
        yaxis=dict(title=y_title)
    )
    return fig


def add_baseline(
        fig: Figure,
        base_data: [float, ArrayLike],
        label: str = 'Baseline'
) -> Figure:
    """
    Add a baseline data to the figure.

    The ``base_data`` may be a list providing a value at each x value
    of the ``fig`` or may be a constant value to generate a line.

    Args:
        fig: The figure to add the baseline data to.
        base_data: The baseline data to add to the figure.
        label: The label of the baseline data.

    Returns:
        The figure updated with the baseline data added.
    """
    x_data = fig.data[0]['x']
    base_data = base_data if isinstance(base_data, list) \
        else np.ones(len(x_data)) * base_data

    fig.add_trace(go.Scatter(
        x=x_data, y=base_data, mode='lines', name=label
    ))

    return fig


def merge(
        figs: list[Figure],
        rows: int = None,
        cols: int = None
) -> Figure:
    """
    Merge a list of plotly.graph_objects.Figure into a single plot.

    The number of rows and columns of the merged plot is determined by
    the number of plots to merge if ``rows`` or ``cols`` are not
    provided. The titles and axis labels of the plots are transferred
    to the merged plot.

    Args:
        figs: the plots to merge.
        rows: the number of rows of the merged plot.
        cols: the number of columns of the merged plot.

    Returns:
        merged_figs: the merged plot.
    """
    num_figs = len(figs)

    if rows is None or cols is None:
        # Determine the number of rows and columns of the merged plot.
        cols = math.ceil(math.sqrt(num_figs))
        rows = math.ceil(num_figs / cols)

    merged_figs = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[fig.layout.title.text for fig in figs])

    for i, fig in enumerate(figs):
        row = (i // cols) + 1
        col = (i % cols) + 1
        for trace in fig.data:
            merged_figs.add_trace(trace, row=row, col=col)

        merged_figs.update_xaxes(
            title_text=fig.layout.xaxis.title.text, row=row, col=col)
        merged_figs.update_yaxes(
            title_text=fig.layout.yaxis.title.text, row=row, col=col)

    return merged_figs
