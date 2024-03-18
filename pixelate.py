import sys, os
from PIL import Image
import time, random, copy

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
        image = Image.open(filename).convert("RGB")
    except FileNotFoundError:
        print("File not found.")
        sys.exit(1)

    filename_only = os.path.splitext(os.path.basename(filename))[0]

    cell_image = CellImage(image, width, height)
    print("discontinuity=", calculate_discontinuity(cell_image))

    pixelated_image = cell_image.extract()
    pixelated_image = Image.fromarray(pixelated_image)
    pixelated_img_path = f"{filename_only}_resized.png"
    pixelated_image.save(pixelated_img_path)

    s = time.time()
    cell_image.set_dominant()
    print("discontinuity=", calculate_discontinuity(cell_image))
    print("set_dominant time=", time.time() - s)

    pixelated_image = cell_image.extract()
    pixelated_image = Image.fromarray(pixelated_image)
    pixelated_img_path = f"{filename_only}_pixelated.png"
    pixelated_image.save(pixelated_img_path)

    #pixelated_image = reduce_colors(pixelated_image, 32)
    #pixelated_img_path = f"{filename_only.split('.')[0]}_pixelated_reduced.png"
    #pixelated_image.save(pixelated_img_path)

if __name__ == "__main__":
    main()