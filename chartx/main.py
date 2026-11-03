import cv2
import matplotlib.pyplot as plt
from chartx.preprocessing import glare


path = "sample-images/IMG_1294.jpg"


def main():
    image = cv2.imread(path)
    if image is None:
        print(f"Error: image at {path} not found")
        exit(1)

    result = glare.clean_whiteboard(image)

    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    main()