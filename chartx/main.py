import cv2
import matplotlib.pyplot as plt
from chartx.preprocessing import glare
from chartx.detection import bar_graph


path = "sample-images/barchart_test.jpg"


from dataclasses import fields


# Display data for a graph or chart
def print_table(rows):
    """Print a list of dataclass objects as an aligned table."""
    if not rows:
        print("(no rows)")
        return

    headers = [f.name for f in fields(rows[0])]
    cells = [["-" if getattr(r, h) is None else str(getattr(r, h)) for h in headers] for r in rows]
    widths = [max(len(h), *(len(row[i]) for row in cells)) for i, h in enumerate(headers)]

    def line(items):
        return "  ".join(item.ljust(w) for item, w in zip(items, widths))

    print(line(headers))
    print(line(["-" * w for w in widths]))
    for row in cells:
        print(line(row))


def main():

    # Read in an image
    image = cv2.imread(path)
    if image is None:
        print(f"Error: image at {path} not found")
        exit(1)

    # Clean up the image
    result = glare.clean_whiteboard(image)

    # Check if this is a bar chart and get its data object list
    bar_graph_data = bar_graph.detect_bar_graph(result)

    if bar_graph_data == None:
        print("No bar graph found")

    # Display the bar chart data
    print_table(bar_graph_data)

    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    main()