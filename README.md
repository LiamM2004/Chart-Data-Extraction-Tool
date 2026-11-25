# chartx

Extract data from photos of real-world charts.

## Pipeline

1. **Preprocessing** *(WIP)*
   - **Straighten** - find the board, fix perspective, round off pies, level the chart (`chartx/preprocessing/deskew/`)
   - **Clean up** - remove glare, balance the whites, sharpen (`chartx/preprocessing/glare.py`)
2. **Axis detection** - find the axes and plot area *(todo)*
3. **Tick reading** - locate ticks and OCR their labels *(todo)*
4. **Calibration** - fit pixel -> value mapping per axis *(todo)*
5. **Extraction** - recover series from lines/points/bars *(WIP: bar charts in `chartx/detection/bar_graph.py`, pie charts in `chartx/detection/pie_graph.py`)*
6. **Output** - CSV/JSON with confidence and warnings *(todo)*

## Setup

```
python3 -m venv .venv && source .venv/bin/activate      # Python 3.10 to 3.13
pip install -e .
```

## Running

From the project root, with the virtual environment active, give the photo, and optionally the
type of chart first:

```
python -m chartx sample-images/barchart_test.jpg        # work out which kind of chart it is
python -m chartx bar sample-images/barchart_test.jpg    # it's a bar chart
python -m chartx pie sample-images/IMG_1294.jpg         # it's a pie chart
```

The photo is straightened and its glare cleaned up, then the chart is read.

Without a chart type (or with `auto`), both detectors run and their answers are checked against
each other (`chartx/detection/auto.py`). On a bar chart the pie detector can mistake the bars for
a pie, so a pie with bars inside its circle is dropped. If a board has both a bar chart and a pie
chart, the bigger one is read and the command to read the other is printed. If no chart is found,
a message says so and the cleaned photo is still shown. Line charts aren't supported yet.

What each chart type prints:

- **bar** (`chartx/detection/bar_graph.py`): each bar is printed as a table row (position, size,
  height relative to the tallest bar, value, colour and the label written under it), and the cleaned photo opens in a
  matplotlib window with a numbered box around each bar. Bar values are measured against the
  top gridline, taken as 10 on an axis that starts at 0.
- **pie** (`chartx/detection/pie_graph.py`): each sector is printed as a table row (angles,
  percentage, the percentage written on it and its label), and the cleaned photo opens in a
  matplotlib window. Reading the writing needs PaddleOCR (`pip install paddleocr`).

If the board can't be found, the photo is used as it is, without straightening.

Pies with coloured sectors and no drawn dividers (like `IMG_1294.jpg`) are first turned into a black
line drawing: every fill gets one label by its hue, and the borders between labels become the dividers.

## Tests

```
python -m unittest discover tests -v
```

Run one file with `-p test_main.py`, or tests whose names match a word with `-k glare`.
The same command runs automatically on every pull request and every push to main.
The pie chart tests (`tests/test_pie_graph.py`) download the PaddleOCR models on the first run.

The project uses OpenCV 4.10.0.84 (`opencv-contrib-python`), because PaddleOCR needs exactly this
version. The deskew thresholds were first tuned with OpenCV 5.0.0.93; all tests also pass on 4.10.
Do not install `opencv-python` next to it: both packages use the same `cv2` folder.
Other OpenCV versions may shift borderline photos slightly.

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
   - *Nothing found with confidence:* the photo is left untouched and flagged. The corners can be
     dragged into place by hand with `select_corners_manually` (`corner_editor.py`), but
     `main.py` doesn't open this window at the moment.

   The board's true width-to-height ratio comes from its corners and the assumed camera
   (Zhang & He, 2004, "Whiteboard scanning and image enhancement").
2. **Round off pies.** A pie's outline should be a circle. If it is still squashed after step 1,
   stretch the photo until it is round (`ellipse.py`).
3. **Level the chart** (`rotation.py`). Bar and line charts are levelled on the "L" of their two
   axes. A pie has no axes, so it is levelled on its writing (title and legend) instead.

**The assumed camera.** Step 1 needs the lens's focal length. The default, 0.72 times the photo's
long side, suits a phone's main camera. For another lens, pass `focal_frac=` to `deskew()` (an
ultra-wide is about 0.4, a 2x zoom about 1.4).

## Layout

```
chartx/
  main.py             run everything: preprocess a photo, find its chart, print its data
  preprocessing/
    deskew/           straighten a photo (see its __init__.py for a map of the files)
    glare.py          glare removal and white balance
  detection/
    auto.py           work out which kind of chart is in a photo
    bar_graph.py      find the bars in a bar chart and read their values
    pie_graph.py      find the sectors of a pie chart and read their labels (PaddleOCR)
tests/                python -m unittest discover tests -v
```
