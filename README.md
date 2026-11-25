# chartx

Extract data from photos of real-world charts.

## Pipeline

1. **Preprocessing** *(WIP)*
   - **Straighten** - find the board, fix perspective, round off pies, level the chart (`chartx/preprocessing/deskew/`)
   - **Clean up** - remove glare, balance the whites, sharpen (`chartx/preprocessing/glare.py`)
2. **Axis detection** - find the axes and plot area (`chartx/detection/axes.py`)
3. **Tick reading** - find the axis labels and OCR them (`chartx/detection/axes.py`)
4. **Calibration** - fit pixel -> value mapping for the y-axis (`chartx/detection/axes.py`)
5. **Extraction** - recover series from lines/bars/pies
   - **Line charts** - `chartx/detection/line_graph.py`
   - **Bar charts** - `chartx/detection/bar_graph.py` *(reads its values off the gridlines, not
     yet off the axis labels that step 3 now provides)*
   - **Pie charts** - `chartx/detection/pie_graph.py` *(not wired into `main.py` yet)*
6. **Output** - CSV/JSON with confidence and warnings *(todo)*

## Setup

```
python3 -m venv .venv && source .venv/bin/activate      # Python 3.10 to 3.13
pip install -e .
```

That is the whole installation. All the text — axis labels and pie sectors alike — is read by
PaddleOCR, which `pip` installs, so there is no separate OCR program to set up. The first run
downloads the PaddleOCR models.

## Running

From the project root, with the virtual environment active, give the photo, and optionally the
type of chart first:

```
python -m chartx sample-images/barchart_test.jpg           # work out which kind of chart it is
python -m chartx bar sample-images/barchart_test.jpg       # it's a bar chart
python -m chartx line sample-images/20261027_121908.jpg    # it's a line chart
python -m chartx pie sample-images/IMG_1294.jpg            # it's a pie chart
```

The photo is straightened and its glare cleaned up, then the chart is read.

Without a chart type (or with `auto`), all three detectors run and their answers are checked
against each other (`chartx/detection/auto.py`). The pie detector can be fooled by the other
charts: it can fit a circle around a bar chart's bars, or take the corner where a line chart's
axes meet for a pie's centre. A real pie never has bars or line points inside it, so a pie that
does is dropped. If a board has two charts side by side, the bigger one is read and the command
to read the other is printed. If no chart is found, a message says so and the cleaned photo is
still shown. `--manual` and `--corners` work without a chart type too.

What each chart type prints:

- **bar** (`chartx/detection/bar_graph.py`): each bar is printed as a table row (position, size,
  height relative to the tallest bar, value, colour and the label written under it), and the
  cleaned photo opens in a matplotlib window with a numbered box around each bar. Bar values are
  measured against the top gridline, taken as 10 on an axis that starts at 0.
- **line** (`chartx/detection/line_graph.py`): each reading is printed as a table row (which line
  it belongs to, the category it sits under, where it is in the photo, its value on the y-axis
  and the colour of the pen it was drawn with), and the cleaned photo opens in a matplotlib
  window.
- **pie** (`chartx/detection/pie_graph.py`): each sector is printed as a table row (angles,
  percentage, the percentage written on it and its label), and the cleaned photo opens in a
  matplotlib window.

Everything depends on the photo being straightened. When that fails the reason is printed; pass
`--manual` to drag the board's four corners into place yourself, or `--corners x,y x,y x,y x,y`
to give them outright.

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
version. The deskew and bar chart thresholds were tuned with OpenCV 5.0.0.93 and also pass on 4.10.

The line chart tests pass on both. They were written against 5.0.0.93, and bringing them up on
4.10 was worth doing: deskew finds slightly different board corners on each version, so the
straightened photo comes out a few pixels different in size, and three of the four failures that
exposed were real bugs that 5.0.0.93 had only been getting away with. Fixing them improved the
readings on 5.0.0.93 as well, from 1.3% mean error to 1.1%.
Do not install `opencv-python` next to it: both packages use the same `cv2` folder.
Other OpenCV versions may shift borderline photos slightly.

Sample chart photos (varied angles and lighting) live in `sample-images/`.

### How accurate the line charts are

`tests/fixtures/line_charts.json` holds the values read off the four line chart photos by hand.
`tests/test_line_graph.py` checks every extracted value against them, and

```
python tests/check_accuracy.py          # or a photo name, e.g. 121850
```

prints the whole comparison, which is the thing to run while changing the detection. All 40
values currently come out within 6% of the axis range, 1.1% on average and 2.9% at worst.

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
    axes.py           find the axes, read their labels, turn pixel rows into values
    bar_graph.py      find the bars in a bar chart and read their values
    line_graph.py     follow the lines in a line chart and read their values
    ocr.py            read handwriting off a label crop (PaddleOCR)
    pie_graph.py      find the sectors of a pie chart and read their labels (PaddleOCR)
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
3. **Read the labels.** Every label crop, on both axes, goes to PaddleOCR on its own. A reading
   that is all digits becomes a candidate value for the y-axis scale; the rest are category text.
   Category labels are then matched against the sequences charts normally use — the months, the
   days of the week, quarters, or an evenly spaced run of numbers. Two labels reading clearly is enough to say which sequence it is
   and where it starts, and the rest are named from their place in it. On the sample photos it
   is the years it repairs; the months read directly.
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

**Category labels**, all 26 of them on the sample photos, read correctly. PaddleOCR gets 23 of
them straight off and the numeric-run fitting repairs the last three, all of which are years
("990", "20102"). It is asked one label crop at a time: given a whole strip in one call it
returns the row as a single box ("1990 1995 2000 ..."), which cannot be split back into one text
per label.

Of the 24 y-axis labels it reads 18, and gets all 18 right, returning nothing for the other six.
Reading nothing is the failure to want here — a chart with too few readings is turned down rather
than reported with wrong values.

A chart whose y-axis labels can't be read at all is not reported as a line chart. There would
be no values to give for it, and in practice what reaches that point is usually not a line
chart at all: the pie photo `IMG_1294.jpg` offers up a passable pair of axes and nine unreadable
labels, and used to come back as a line chart with 27 empty rows.

**What else was tried.** Measured against the fixtures, as mean error over the 40 hand-read
values (the current method comes in at 1.1%):

| way of following the lines | mean error | notes |
| --- | --- | --- |
| one row per column, the middle of the ink | 22.7% | only ever finds one line |
| nearest run, ignoring which way the line was heading | 9.8% | loses both lines at a crossing |
| least-bending whole path (dynamic programming) | 16.8% | see below |
| the same, with a limit on how far one step may jump | 7.1% | drops a line at the crossing |
| nearest run to where the line was heading (current) | 1.1% | |

Dynamic programming is the obvious way to pick a whole path at once, and it is in the plan this
work started from, but it does not suit a chart with more than one line. With the state as the
pair (previous run, current run) the path forgets which line it set out on after a single step,
so every line collapses onto the one route that bends least overall and hops between lines at
each crossing. Constraining the step keeps them apart but then a line simply ends where it meets
another. Following each line forwards from where it has last been seen avoids both.

Straightening each label on its own slant before reading it was also tried, since the categories
are written on a slant and the y-axis numbers are not. It gained one label out of 26 and cost
twice the OCR, so it is not in: the slanted labels read correctly without it.

Categories that are an evenly spaced run of numbers (1, 2, 3, 4) *are* read, the same way the
months are: each pair of readings that parse as numbers proposes a first value and a step, and
the best-fitting run names the labels the OCR got wrong. It is what turns 23 of the 26 sample
labels into 26 of 26, repairing the years the OCR fumbles.
