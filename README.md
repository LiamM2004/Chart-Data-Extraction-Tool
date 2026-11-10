# chartx

Extract data from photos of real-world charts.

## Pipeline

1. **Preprocessing** - fix perspective and level the axes (`chartx/preprocessing/deskew.py`) *(WIP)*
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
python -m chartx.preprocessing.deskew tests/data/photo.jpg -o out/photo.png --debug out/debug.png
```

Put sample chart photos (varied angles and lighting) in `tests/data/`.
