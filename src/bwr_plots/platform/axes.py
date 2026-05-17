from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _get_scale_and_suffix(max_value: float) -> tuple[float, str]:
    abs_max = abs(max_value) if pd.notna(max_value) else 0
    if abs_max >= 1_000_000_000:
        return 1_000_000_000, "B"
    if abs_max >= 1_000_000:
        return 1_000_000, "M"
    if abs_max >= 1_000:
        return 1_000, "K"
    return 1, ""


def _nice_number(value: float, round_: bool = False) -> float:
    exp = math.floor(np.log10(value))
    f = value / 10**exp
    if round_:
        if f < 1.5:
            nf = 1
        elif f < 3:
            nf = 2
        elif f < 7:
            nf = 5
        else:
            nf = 10
    else:
        if f <= 1:
            nf = 1
        elif f <= 2:
            nf = 2
        elif f <= 5:
            nf = 5
        else:
            nf = 10
    return nf * 10**exp


def calculate_yaxis_grid_params(
    y_data,
    padding: float = 0.05,
    num_gridlines: int = 5,
    top_extra: float = 0.002,
) -> dict[str, object]:
    y_data = np.asarray(y_data)
    y_min_data = float(np.nanmin(y_data))
    y_max = float(np.nanmax(y_data))
    if y_min_data == y_max:
        y_max = y_min_data + 1

    data_range = y_max - y_min_data
    initial_axis_min = 0 if y_min_data >= 0 else y_min_data - data_range * padding
    initial_axis_max = y_max + data_range * padding
    raw_tick = (initial_axis_max - initial_axis_min) / (num_gridlines - 1)
    dtick = _nice_number(raw_tick, round_=True)
    snapped_axis_min = np.floor(initial_axis_min / dtick) * dtick
    final_axis_min = 0.0 if y_min_data >= 0 and snapped_axis_min < 0 else snapped_axis_min
    n_ticks = int(np.ceil((y_max - final_axis_min) / dtick)) + 1
    final_axis_max = final_axis_min + dtick * (n_ticks - 1)
    final_axis_max_extended = final_axis_max + (final_axis_max - final_axis_min) * top_extra
    return {
        "range": [final_axis_min, final_axis_max_extended],
        "tick0": final_axis_min,
        "dtick": dtick,
        "tickmode": "linear",
    }


def add_top_gridline(
    fig,
    y_max,
    gridline_color: str = "#404040",
    gridline_width: float = 1.5,
    gridline_dash: str = "solid",
) -> None:
    fig.add_shape(
        type="line",
        xref="x",
        yref="y",
        x0=fig.layout.xaxis.range[0] if fig.layout.xaxis.range else 0,
        x1=fig.layout.xaxis.range[1] if fig.layout.xaxis.range else 1,
        y0=y_max,
        y1=y_max,
        line=dict(
            color=gridline_color,
            width=gridline_width,
            dash=gridline_dash,
        ),
        layer="below",
    )


def add_top_gridline_paper(
    fig,
    gridline_color: str = "#404040",
    gridline_width: float = 1.5,
    gridline_dash: str = "solid",
) -> None:
    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=0,
        x1=1,
        y0=1,
        y1=1,
        line=dict(
            color=gridline_color,
            width=gridline_width,
            dash=gridline_dash,
        ),
        layer="below",
    )


def ensure_datetime_index(
    _plotter,
    data: Union[pd.DataFrame, pd.Series],
    xaxis_is_date: Optional[bool] = True,
) -> Union[pd.DataFrame, pd.Series]:
    if data is None or data.empty or xaxis_is_date is False:
        return data
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            original_name = data.index.name
            data_copy = data.copy()
            data_copy.index = pd.to_datetime(data_copy.index, errors="raise")
            data_copy.index.name = original_name
            if isinstance(data_copy.index, pd.DatetimeIndex) and data_copy.index.tz is not None:
                data_copy.index = data_copy.index.tz_localize(None)
            return data_copy
        except Exception as exc:
            print(f"[WARNING] _ensure_datetime_index: Could not convert index to datetime: {exc}.")
            return data

    try:
        if data.index.tz is not None:
            data_copy = data.copy()
            data_copy.index = data_copy.index.tz_localize(None)
            return data_copy
    except Exception:
        pass
    return data


def prepare_xaxis_data(
    plotter,
    data: Union[pd.DataFrame, pd.Series],
    xaxis_is_date: bool,
) -> Union[pd.DataFrame, pd.Series]:
    if data is None or data.empty:
        return data
    if xaxis_is_date:
        return ensure_datetime_index(plotter, data, xaxis_is_date=True)
    if pd.api.types.is_numeric_dtype(data.index.dtype):
        try:
            data_copy = data.copy()
            data_copy.index = data_copy.index.astype(str)
            return data_copy
        except Exception as exc:
            print(f"Warning: Failed to convert numeric index to string in _prepare_xaxis_data: {exc}")
    return data


def apply_common_axes(
    plotter,
    fig: go.Figure,
    axis_options: Optional[dict] = None,
    is_secondary: bool = False,
    axis_min_calculated: Optional[float] = None,
    xaxis_is_date: Optional[bool] = True,
) -> None:
    cfg_axes = plotter.config["axes"]
    default_opts = {
        "primary_title": cfg_axes["y_primary_title_text"],
        "secondary_title": cfg_axes["y_secondary_title_text"],
        "primary_prefix": cfg_axes["y_primary_tickprefix"],
        "secondary_prefix": cfg_axes["y_secondary_tickprefix"],
        "primary_suffix": cfg_axes["y_primary_ticksuffix"],
        "secondary_suffix": cfg_axes["y_secondary_ticksuffix"],
        "primary_range": cfg_axes["y_primary_range"],
        "secondary_range": cfg_axes["y_secondary_range"],
        "primary_tickformat": cfg_axes["y_primary_tickformat"],
        "secondary_tickformat": cfg_axes["y_secondary_tickformat"],
        "primary_hoverformat": cfg_axes.get("y_primary_hoverformat"),
        "secondary_hoverformat": cfg_axes.get("y_secondary_hoverformat"),
        "x_tickformat": cfg_axes["x_tickformat"],
        "x_hoverformat": cfg_axes.get("x_hoverformat"),
        "x_nticks": cfg_axes["x_nticks"],
        "x_range": None,
        "x_title_text": cfg_axes["x_title_text"],
    }
    merged_options = default_opts.copy()
    if axis_options:
        merged_options.update(axis_options)

    xaxis_type = merged_options.get("x_type", "category" if xaxis_is_date is False else "date")
    xaxis_tickformat = "" if xaxis_type == "category" else merged_options.get("x_tickformat", cfg_axes["x_tickformat"])
    xaxis_tickformatstops = (
        None
        if xaxis_type != "date"
        else merged_options.get("x_tickformatstops", cfg_axes.get("x_tickformatstops"))
    )

    fig.update_xaxes(
        type=xaxis_type,
        title=dict(text=merged_options.get("x_title_text", cfg_axes["x_title_text"]), font=plotter._get_font_dict("axis_title")),
        showline=True,
        linewidth=cfg_axes.get("gridwidth", 2.5),
        linecolor=cfg_axes.get("y_gridcolor", "rgb(38, 38, 38)"),
        tickcolor=cfg_axes["y_gridcolor"],
        showgrid=cfg_axes["showgrid_x"],
        gridcolor=cfg_axes["x_gridcolor"],
        gridwidth=cfg_axes.get("gridwidth", 1),
        ticks="outside",
        tickwidth=cfg_axes["tickwidth"] * 1.5,
        ticklen=cfg_axes["x_ticklen"],
        ticklabelstandoff=0,
        nticks=merged_options["x_nticks"],
        tickformat=xaxis_tickformat,
        tickformatstops=xaxis_tickformatstops,
        hoverformat=merged_options.get("x_hoverformat"),
        tickfont=plotter._get_font_dict("tick"),
        zeroline=False,
        zerolinewidth=0,
        zerolinecolor="rgba(0,0,0,0)",
        showspikes=cfg_axes["showspikes"],
        spikethickness=cfg_axes["spikethickness"],
        spikedash=cfg_axes["spikedash"],
        spikecolor=cfg_axes["spikecolor"],
        spikemode=cfg_axes["spikemode"],
        showticklabels=True,
        tickmode="auto",
        range=merged_options["x_range"],
        visible=True,
        color="rgba(0,0,0,0)",
        anchor="free",
        position=0,
        fixedrange=True,
        tickvals=merged_options.get("x_tickvals", None),
    )

    primary_tickformat = merged_options["primary_tickformat"]
    primary_dtick = merged_options.get("primary_dtick", None)
    primary_tick0 = merged_options.get("primary_tick0", None)
    primary_tickmode = "linear" if primary_dtick is not None else merged_options.get("primary_tickmode", "auto")
    if primary_dtick is not None and isinstance(primary_dtick, (float, int)) and primary_dtick % 1 != 0:
        if primary_tickformat in [",d", ",.0f", "d", ".0f"]:
            primary_tickformat = ",.2f"

    fig.update_yaxes(
        title=dict(text=merged_options["primary_title"], font=plotter._get_font_dict("axis_title")),
        tickprefix=merged_options["primary_prefix"],
        ticksuffix=merged_options["primary_suffix"],
        tickfont=plotter._get_font_dict("tick"),
        showgrid=cfg_axes["showgrid_y"],
        gridcolor=cfg_axes["y_gridcolor"],
        gridwidth=cfg_axes.get("gridwidth", 1),
        range=merged_options["primary_range"],
        tickformat=primary_tickformat,
        hoverformat=merged_options.get("primary_hoverformat"),
        secondary_y=False,
        linecolor=cfg_axes["linecolor"],
        tickcolor="rgba(0,0,0,0)",
        ticks="",
        tickwidth=0,
        showline=False,
        linewidth=cfg_axes["linewidth"],
        zeroline=False,
        zerolinewidth=0,
        zerolinecolor="rgba(0,0,0,0)",
        showticklabels=True,
        tickmode=primary_tickmode,
        tick0=primary_tick0,
        dtick=primary_dtick,
        ticklen=0,
        fixedrange=True,
    )

    if is_secondary:
        secondary_tickformat = merged_options["secondary_tickformat"]
        secondary_dtick = merged_options.get("secondary_dtick", None)
        secondary_tick0 = merged_options.get("secondary_tick0", None)
        secondary_tickmode = "linear" if secondary_dtick is not None else merged_options.get("secondary_tickmode", "auto")
        if secondary_dtick is not None and isinstance(secondary_dtick, (float, int)) and secondary_dtick % 1 != 0:
            if secondary_tickformat in [",d", ",.0f", "d", ".0f"]:
                secondary_tickformat = ",.2f"

        fig.update_yaxes(
            title=dict(text=merged_options["secondary_title"], font=plotter._get_font_dict("axis_title")),
            tickprefix=merged_options["secondary_prefix"],
            ticksuffix=merged_options["secondary_suffix"],
            tickfont=plotter._get_font_dict("tick"),
            showgrid=False,
            gridcolor=cfg_axes["y_gridcolor"],
            gridwidth=cfg_axes.get("gridwidth", 1),
            range=merged_options["secondary_range"],
            tickformat=secondary_tickformat,
            hoverformat=merged_options.get("secondary_hoverformat"),
            secondary_y=True,
            linecolor=cfg_axes["linecolor"],
            tickcolor="rgba(0,0,0,0)",
            ticks="",
            tickwidth=0,
            showline=False,
            linewidth=cfg_axes["linewidth"],
            zeroline=False,
            zerolinewidth=cfg_axes["zerolinewidth"],
            zerolinecolor=cfg_axes["zerolinecolor"],
            showticklabels=True,
            tickmode=secondary_tickmode,
            tick0=secondary_tick0,
            dtick=secondary_dtick,
            ticklen=0,
            fixedrange=True,
        )
