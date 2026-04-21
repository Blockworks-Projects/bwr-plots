---
name: "render-chart-cli"
description: "Use when the user wants the exact bwr-plots CLI command for rendering a chart from CSV or Excel, including available chart kinds, CLI flags, chart spec fields, and supported layer metadata."
metadata:
  short-description: "Render charts with the CLI"
---

# Render Chart CLI

Use this skill when the task is to render a chart through the `bwr-plots` CLI instead of the Python API.

Source of truth:
- `src/bwr_plots/cli/main.py`
- `src/bwr_plots/platform/specs.py`
- `src/bwr_plots/platform/registry.py`

## Discover chart kinds

```bash
cd /Users/daniel/Developer/bwr-plots
uv run bwr-plots list-charts
```

Available chart kinds right now:

| kind | display name | supports layers |
| --- | --- | --- |
| `bar` | `Bar` | yes |
| `horizontal_bar` | `Horizontal Bar` | yes |
| `metric_share_area` | `Metric Share Area` | yes |
| `multi_bar` | `Multi Bar` | yes |
| `pie` | `Pie` | no |
| `point` | `Point` | yes |
| `scatter` | `Scatter / Line` | yes |
| `stacked_bar` | `Stacked Bar` | yes |

## Core command

```bash
cd /Users/daniel/Developer/bwr-plots
uv run bwr-plots render \
  --chart scatter \
  --data /absolute/path/input.csv \
  --output-file /absolute/path/output.html \
  --spec-json '{"title":"Chart Title","subtitle":"Subtitle","source":"Blockworks Research"}' \
  --date-col date \
  --include-plotlyjs cdn \
  --open
```

`--spec-json` and `--spec-file` are interchangeable. If both are passed, inline JSON wins.

## CLI flags

| flag | required | notes |
| --- | --- | --- |
| `--chart` | yes | Registered chart kind. |
| `--data` | yes | CSV, XLS, XLSX, or XLSM input. |
| `--output-file` | yes | HTML output path. `.html` is added if missing. |
| `--spec-json` | no | Inline JSON object for the chart spec. |
| `--spec-file` | no | Path to a JSON file containing the chart spec object. |
| `--layers-json` | no | Inline JSON array of layer specs. |
| `--layers-file` | no | Path to a JSON file containing a layer spec array. |
| `--sheet` | no | Excel sheet name. Ignored for CSV. |
| `--index-column` | no | Set this column as the index if no date index is chosen. |
| `--date-col` | no | Parse this column as datetimes and use it as the index. |
| `--include-plotlyjs` | no | `cdn` or `inline`. Default: `cdn`. |
| `--open` / `--no-open` | no | Open the output HTML in a browser. Default: open. |

If `--date-col` is omitted and the input has exactly one date-like column, the CLI will auto-detect it and use it as the index.

## Spec JSON shape

`--spec-json` and `--spec-file` must be a JSON object with `kind` implied by `--chart`.

### Common chart fields

These fields are available on every chart kind.

| key | type | default |
| --- | --- | --- |
| `preset` | `string \| null` | `null` |
| `title` | `string` | `""` |
| `subtitle` | `string` | `""` |
| `source` | `string` | `""` |
| `date` | `string \| null` | `null` |
| `prefix` | `string \| null` | `null` |
| `suffix` | `string \| null` | `null` |
| `width` | `integer \| null` | `null` |
| `height` | `integer \| null` | `null` |
| `use_watermark` | `boolean \| null` | `null` |
| `x_axis_title` | `string \| null` | `null` |
| `y_axis_title` | `string \| null` | `null` |
| `axis_options` | `object \| null` | `null` |
| `legend_order` | `string[] \| null` | `null` |
| `series_colors` | `object \| null` | `null` |
| `config_override` | `object` | `{}` |

Notes:
- `axis_options` is the generic axis override object.
- `config_override` is the generic layout/config override object.
- `series_colors` maps series names to hex colors.
- `legend_order` is an ordered array of legend labels.

### Chart-specific fields

| chart | extra fields |
| --- | --- |
| `bar` | `bar_color`, `show_legend` |
| `horizontal_bar` | `y_column`, `x_column`, `show_bar_values`, `color_positive`, `color_negative`, `sort_ascending`, `bar_height`, `bargap` |
| `metric_share_area` | `xaxis_is_date`, `show_legend`, `smoothing_window` |
| `multi_bar` | `xaxis_is_date`, `show_legend`, `colors`, `scale_values`, `show_bar_values`, `tick_frequency` |
| `pie` | `show_values`, `text_position`, `hole_size`, `show_legend` |
| `point` | `x_column`, `y_column`, `group_column`, `label_column`, `size_column`, `xaxis_is_date`, `show_legend`, `marker_size`, `marker_opacity`, `uniform_color`, `show_trendline`, `trendline_type`, `trendline_color`, `show_r_squared` |
| `scatter` | `xaxis_is_date`, `show_legend`, `fill_mode`, `fill_color`, `smoothing_window`, `auto_scale_y_values`, `secondary_y_data`, `secondary_y_prefix`, `secondary_y_suffix` |
| `stacked_bar` | `xaxis_is_date`, `show_legend`, `colors`, `scale_values`, `sort_descending` |

Important CLI note:
- `scatter.secondary_y_data` exists on the spec type, but it expects a pandas object and is not a practical CLI JSON field. Treat it as API-oriented, not CLI-oriented.

### Field details that are easy to miss

| key | type | default | notes |
| --- | --- | --- | --- |
| `show_legend` | `boolean` | varies by chart | `bar` defaults `false`; most others default `true`. |
| `xaxis_is_date` | `boolean` | usually `true` | Present on date-oriented chart types. |
| `colors` | `object \| null` | `null` | Present on `multi_bar` and `stacked_bar`. |
| `text_position` | `"inside" \| "outside" \| "auto" \| null` | `null` | Pie only. |
| `trendline_type` | `string` | `"linear"` | Point only. |
| `uniform_color` | `boolean` | `false` | Point only. |
| `auto_scale_y_values` | `boolean` | `true` | Scatter only. |
| `show_bar_values` | `boolean` | chart-specific | `horizontal_bar` defaults `true`; `multi_bar` is optional. |

## Layer JSON shape

Only charts with `supports layers = yes` can use `--layers-json` or `--layers-file`.

Current layer kinds:

| kind | purpose |
| --- | --- |
| `highlight_bands` | Overlay translucent vertical highlight bands. |

Layer payload example:

```bash
uv run bwr-plots render \
  --chart scatter \
  --data /absolute/path/input.csv \
  --output-file /absolute/path/output.html \
  --spec-json '{"title":"BTC Price","source":"Blockworks Research"}' \
  --layers-json '[{"kind":"highlight_bands","bands":[{"start":"2024-01-01","end":"2024-03-31","label":"Q1","color":"#5637cd","opacity":0.16}]}]'
```

`highlight_bands` layer fields:

| key | type | default |
| --- | --- | --- |
| `bands` | `array` | `[]` |

Each band object supports:

| key | type | default |
| --- | --- | --- |
| `start` | `string \| number` | required |
| `end` | `string \| number` | required |
| `label` | `string \| null` | `null` |
| `color` | `string` | `"#5637cd"` |
| `opacity` | `number` | `0.16` |
| `line_color` | `string \| null` | `null` |
| `annotation_position` | `"top left" \| "top right" \| "bottom left" \| "bottom right"` | `"top left"` |

## Minimal spec examples

Scatter:

```json
{
  "title": "BTC Price",
  "subtitle": "Daily close",
  "source": "Blockworks Research",
  "show_legend": true,
  "series_colors": {
    "BTC": "#5637cd"
  }
}
```

Horizontal bar:

```json
{
  "title": "Protocol Revenue",
  "source": "Blockworks Research",
  "x_column": "revenue",
  "y_column": "protocol",
  "show_bar_values": true,
  "color_positive": "#5637cd",
  "color_negative": "#ef798a",
  "sort_ascending": false
}
```
