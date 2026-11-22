# chartx

Extract data from photos of real-world charts.

## Pipeline

1. **Preprocessing** *(WIP)*
   - **Straighten** - find the board, fix perspective, round off pies, level the chart (`chartx/preprocessing/deskew/`)
   - **Clean up** - remove glare, balance the whites, sharpen (`chartx/preprocessing/glare.py`)
2. **Axis detection** - find the axes and plot area (`chartx/detection/axes.py`)
3. **Tick reading** - find the axis labels and OCR them (`chartx/detection/axes.py`)
4. **Calibration** - fit pixel -> value mapping for the y-axis (`chartx/detection/axes.py`)
5. **Extraction** - recover series from lines/bars
   - **Line charts** - `chartx/detection/line_graph.py`
   - **Bar charts** - `chartx/detection/bar_graph.py` *(reads its values off the gridlines, not
     yet off the axis labels that step 3 now provides)*
6. **Output** - CSV/JSON with confidence and warnings *(todo)*

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Reading the axis labels needs the **Tesseract** program as well as the `pytesseract` package
that `pip` installs:

```
brew install tesseract            # macOS
sudo apt-get install tesseract-ocr  # Debian/Ubuntu
```

Without it the charts are still found and measured, but the y-axis has no numbers on it, so
values come back empty.

## Running

From the project root, with the virtual environment active:

```
python -m chartx.main sample-images/20261027_121908.jpg
```

This straightens the photo, cleans up its glare, then works out what kind of chart it is: bars
first, and a line chart if there are no bars. What it read is printed as a table and the cleaned
photo opens in a matplotlib window.

```
Found a line chart
line  category  x    y    value   colour
----  --------  ---  ---  ------  -------
1     Jan       518  958  117.8   #000000
1     Feb       621  870  136.75  #000000
...
```

For a line chart each row is one reading: which line it belongs to, the category it sits under,
where it is in the photo, its value on the y-axis and the colour of the pen it was drawn with.
For a bar chart each row is a bar, with its position, size, height relative to the tallest bar,
value and colour.

If the board can't be found, the photo is used as it is, without straightening.

## Tests

```
python -m unittest discover tests -v
```

Run one file with `-p test_main.py`, or tests whose names match a word with `-k glare`.
The same command runs automatically on every pull request and every push to main.

The detection thresholds were tuned with OpenCV 5.0.0.93, the version the automatic tests run on;
other versions may shift borderline photos slightly.

Sample chart photos (varied angles and lighting) live in `sample-images/`.

### How accurate the line charts are

`tests/fixtures/line_charts.json` holds the values read off the four line chart photos by hand.
`tests/test_line_graph.py` checks every extracted value against them, and

```
python tests/check_accuracy.py          # or a photo name, e.g. 121850
```

prints the whole comparison, which is the thing to run while changing the detection. All 40
values currently come out within 6% of the axis range, 1.3% on average.

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
  main.py             run everything: preprocess a photo, detect a bar chart, print its bars
  preprocessing/
    deskew/           straighten a photo (see its __init__.py for a map of the files)
    glare.py          glare removal and white balance
  detection/
    axes.py           find the axes, read their labels, turn pixel rows into values
    bar_graph.py      find the bars in a bar chart and read their values
    line_graph.py     follow the lines in a line chart and read their values
tests/                python -m unittest discover tests -v
  fixtures/           values read off the sample photos by hand
  check_accuracy.py   print how close the extraction gets to them
```

## How line charts are read

`chartx/detection/axes.py` works out where the chart is and what its numbers mean, and
`chartx/detection/line_graph.py` reads the lines. Both expect a preprocessed photo.

1. **Find the axes.** The axes are the longest upright and level strokes that meet in an "L",
   with the corner at the bottom of one and the left end of the other. That corner rule is what
   tells them from the board's own edges, which are longer but meet out in a corner of the photo.
2. **Find the writing.** The strip left of the y-axis and the strip below the x-axis are cut into
   labels: glare specks are dropped, letters are joined into words, and words that sit far closer
   together than labels ever do are joined again. A box twice the usual width is two labels
   written touching, so it is split back in half.
3. **Read the labels.** Each label is put through Tesseract five ways — different crops, sizes and
   page modes — and every reading is kept. Handwriting reads differently at different sizes, and a
   "4" drawn with an open top is a "9" to Tesseract, so committing to one reading loses labels
   that a later step could have used.
   Category labels are then matched against the sequences charts normally use — the months, the
   days of the week, quarters. Two labels reading clearly is enough to say which sequence it is
   and where it starts, and the rest are named from their place in it. That is what turns
   "Sen, Feb, Mar, Apr, flosy, So" into "Jan ... Jun".
4. **Fit the y-axis.** Axis labels climb evenly, so the right scale is the one that puts the most
   labels where it predicts them. Every reading of every label is tried, which lets a label that
   was read three different ways be used through whichever reading agrees with its neighbours,
   and leaves the misreadings out.
5. **Follow the lines.** In each column the ink falls into runs, usually one per line, and the
   lines are followed end to end, each claiming the run nearest to where it was heading. Keeping
   track of where a line was heading, not just where it was, is what carries it straight through
   a crossing: at the meeting point both lines read the same run, and that column is left out of
   both their run-ups so neither forgets which way it was going.
6. **Read the values off.** A hand-drawn line is straight between its readings and bends at them,
   so each category is read at the nearest bend of the line rather than under the middle of its
   label. On these photos the labels are written at a slant, so their middles sit well to one side
   of the points they belong to.

**Known limits.** Category labels that aren't a familiar sequence come back poorly on slanted
handwriting: the years on the chicken-and-fish charts read as "1440, vs, 100, 406, ...". Their
*positions* are right, which is what the values depend on, so only the names suffer. This is a
limit of Tesseract on handwriting rather than of the cropping: over 216 combinations of crop,
scale, thickening and page mode, the union of everything Tesseract returns contains the right
answer for only 8 of the 26 category labels on the sample photos, and just 2 of the 14 years.
Fitting a numeric run to readings that poor is worse than useless — the true sequence
(1990, 1995 ... 2020) comes 2381st out of 16000 candidates — so nothing is inferred there.

A chart whose y-axis labels can't be read at all still has its lines followed, but its values
come back empty.

**What else was tried.** Measured against the fixtures, as mean error over the 40 hand-read
values (the current method comes in at 1.3%):

| way of following the lines | mean error | notes |
| --- | --- | --- |
| one row per column, the middle of the ink | 22.7% | only ever finds one line |
| nearest run, ignoring which way the line was heading | 9.8% | loses both lines at a crossing |
| least-bending whole path (dynamic programming) | 16.8% | see below |
| the same, with a limit on how far one step may jump | 7.1% | drops a line at the crossing |
| nearest run to where the line was heading (current) | 1.3% | |

Dynamic programming is the obvious way to pick a whole path at once, and it is in the plan this
work started from, but it does not suit a chart with more than one line. With the state as the
pair (previous run, current run) the path forgets which line it set out on after a single step,
so every line collapses onto the one route that bends least overall and hops between lines at
each crossing. Constraining the step keeps them apart but then a line simply ends where it meets
another. Following each line forwards from where it has last been seen avoids both.

Straightening each label on its own slant before reading it was also tried, since the categories
are written on a slant and the y-axis numbers are not. It gains one label out of 26 and costs
twice the OCR, so it is not in. Reading the categories as an evenly spaced run of numbers gains
nothing at all: too few of the years are ever read clearly enough to say where the run starts.
