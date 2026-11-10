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

The detection thresholds were tuned with OpenCV 5.0.0.93, the version the automatic tests run on;
other versions may shift borderline photos slightly.

Sample chart photos (varied angles and lighting) live in `sample-images/`.

## How straightening works

The photo goes through three steps. Each is measured on a quick preview and the corrections are
combined, so the photo is resampled only once, which keeps it sharper.

1. **Fix perspective.** Find the whiteboard and warp it back to a rectangle.
   - *Whole board in shot:* look for four sides that all sit on strong edges (`board.py`).
   - *Far side or corners out of shot:* use three edges: top, bottom and one side
     (`three_edge_board.py`). If the top and bottom edges converge by under 10 degrees, the sides
     are taken as parallel ("partial"). At 10 degrees or more, the board's tilt is worked out from
     where they converge, using an assumed phone camera ("steep").
   - *Nothing found with confidence:* the photo is left untouched and flagged. A window opens to
     drag the corners into place (`corner_editor.py`).

   The board's true width-to-height ratio comes from its corners and the assumed camera
   (Zhang & He, 2004, "Whiteboard scanning and image enhancement").
2. **Round off pies.** A pie's outline should be a circle. If it is still squashed after step 1,
   stretch the photo until it is round (`ellipse.py`).
3. **Level the chart** (`rotation.py`). Bar and line charts are levelled on the "L" of their two
   axes. A pie has no axes, so it is levelled on its writing (title and legend) instead.

**The assumed camera.** Step 1 needs the lens's focal length. The default, 0.72 times the photo's
long side, suits a phone's main camera. For another lens pass `--focal` (an ultra-wide is about 0.4,
a 2x zoom about 1.4).

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
