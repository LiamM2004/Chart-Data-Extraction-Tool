# chartx

Extract data from photos of real-world charts.

## Pipeline

1. **Preprocessing** *(WIP)*
   - **Straighten** - find the board, fix perspective, round off pies, level the chart (`chartx/preprocessing/deskew/`)
   - **Clean up** - remove glare, balance the whites, sharpen (`chartx/preprocessing/glare.py`)
2. **Axis detection** - find the axes and plot area *(todo)*
3. **Tick reading** - locate ticks and OCR their labels *(todo)*
4. **Calibration** - fit pixel -> value mapping per axis *(todo)*
5. **Extraction** - recover series from lines/points/bars *(todo)*
6. **Output** - CSV/JSON with confidence and warnings *(todo)*

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Example

```
python -m chartx sample-images/new_3.jpg -o out/photo.png --debug out/debug.png
```

If the board can't be found with confidence, a window opens to drag its four corners into place
(`--no-prompt` skips it for batch runs, `--manual` forces it). `--no-glare` stops after straightening,
and `--focal` sets the lens (default: a phone's main camera). Run `python -m chartx --help` for all options.

Sample chart photos (varied angles and lighting) live in `sample-images/`.

## Layout

```
chartx/
  main.py             the command line (python -m chartx)
  pipeline.py         the processing steps in order; later stages chain on here
  preprocessing/
    deskew/           straighten a photo (see its __init__.py for a map of the files)
    glare.py          glare removal and white balance
tests/                python3 -m unittest discover tests
```
