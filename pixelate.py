import sys
import os
import cv2
import time

from cell import CellImage, calculate_discontinuity, reduce_colors

def main():
    if len(sys.argv) < 3:
        print("Usage: ./pixelate.py filename.png width [height]")
        sys.exit(1)

    filename = sys.argv[1]
    width = int(sys.argv[2])
    if len(sys.argv) > 3:
        height = int(sys.argv[3])
    else:
        height = None

    try:
        image = cv2.imread(filename)
        if image is None:
            raise FileNotFoundError
    except FileNotFoundError:
        print("File not found.")
        sys.exit(1)

    cell_image = CellImage(image, width, height)
    print('discontinuity=', calculate_discontinuity(cell_image))

    pixelated_image = cell_image.extract()
    cv2.imwrite("resized_image.png", pixelated_image)

    s = time.time()
    cell_image.set_dominant()
    print('discontinuity=', calculate_discontinuity(cell_image))
    print('set_dominant time=', time.time()-s)

    pixelated_image = cell_image.extract()
    cv2.imwrite("pixelated_image.png", pixelated_image)

    #pixelated_image = reduce_colors(pixelated_image, 32)
    #cv2.imwrite("pixelated_image_reduced.png", pixelated_image)

if __name__ == "__main__":
    main()
