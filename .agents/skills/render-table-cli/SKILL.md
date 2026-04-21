---
name: "render-table-cli"
description: "Use when the user wants the exact bwr-plots CLI command for rendering a branded table from CSV or Excel, including CLI flags and the full column format schema accepted by render-table."
metadata:
  short-description: "Render tables with the CLI"
---

# Render Table CLI

Use this skill when the task is to render a branded table artifact through the `bwr-plots` CLI.

Source of truth:
- `src/bwr_plots/cli/main.py`
- `src/bwr_plots/api/tables.py`
- `src/bwr_plots/features/tables/types.py`

## Core command

```bash
cd /Users/daniel/Developer/bwr-plots
uv run bwr-plots render-table \
  --data /absolute/path/input.csv \
  --output-file /absolute/path/output.html \
  --title "Treasury Dashboard" \
  --subtitle "Public company snapshot" \
  --source-note "Blockworks Research" \
  --theme dark \
  --logo \
  --column-formats-json '{"nav":{"kind":"currency","notation":"compact","prefix":"$"},"mnav":{"kind":"percent","decimals":1,"suffix":"x"}}' \
  --open
```

`--column-formats-json` and `--column-formats-file` are interchangeable. If both are passed, inline JSON wins.

## CLI flags

| flag | required | notes |
| --- | --- | --- |
| `--data` | yes | CSV, XLS, XLSX, or XLSM input. |
| `--output-file` | yes | HTML output path. `.html` is added if missing. |
| `--title` | no | Table title. |
| `--subtitle` | no | Table subtitle. Default: empty string. |
| `--source-note` | no | Footer source note. Default: empty string. |
| `--theme` | no | `dark` or `light`. Default: `dark`. |
| `--logo` / `--no-logo` | no | Include the packaged BWR logo. Default: logo on. |
| `--sheet` | no | Excel sheet name. Ignored for CSV. |
| `--column-formats-json` | no | Inline JSON object keyed by column name. |
| `--column-formats-file` | no | Path to a JSON file keyed by column name. |
| `--open` / `--no-open` | no | Open the output HTML in a browser. Default: open. |

## Column format JSON shape

`--column-formats-json` and `--column-formats-file` must be a JSON object keyed by source column name.

Example:

```json
{
  "nav": {
    "kind": "currency",
    "notation": "compact",
    "decimals": 1,
    "prefix": "$",
    "suffix": ""
  },
  "ownership_pct": {
    "kind": "percent",
    "notation": "plain",
    "decimals": 2,
    "suffix": "%"
  },
  "ticker": {
    "kind": "text"
  }
}
```

## Column format fields

Each column format object supports these fields:

| key | type | default |
| --- | --- | --- |
| `kind` | `"currency" \| "number" \| "percent" \| "integer" \| "text"` | `"text"` |
| `notation` | `"plain" \| "compact"` | `"plain"` |
| `decimals` | `integer \| null` | `null` |
| `prefix` | `string` | `""` |
| `suffix` | `string` | `""` |

Practical notes:
- `kind` controls the base formatter.
- `notation: "compact"` produces short-number formatting like `1.2M`.
- `decimals` overrides the formatter precision when supported.
- `prefix` and `suffix` are appended after formatting.
- Omit a column from the JSON if it should use the default text rendering.
- Boolean cells stay textual in the rendered artifact.

## Minimal command variants

CSV input:

```bash
uv run bwr-plots render-table \
  --data /absolute/path/table.csv \
  --output-file /absolute/path/table.html \
  --title "Treasury Dashboard" \
  --source-note "Blockworks Research"
```

Excel input with a sheet:

```bash
uv run bwr-plots render-table \
  --data /absolute/path/table.xlsx \
  --sheet "Summary" \
  --output-file /absolute/path/table.html \
  --title "Treasury Dashboard" \
  --theme light \
  --no-logo \
  --column-formats-file /absolute/path/column-formats.json \
  --no-open
```

## Recommended pattern

For anything beyond a tiny one-off command:

1. Keep the dataset in CSV or Excel.
2. Put column formats in a JSON file.
3. Run `uv run bwr-plots render-table ... --column-formats-file ...`.

That keeps the command short and makes formatting changes diffable.
