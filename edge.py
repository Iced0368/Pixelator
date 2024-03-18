import numpy as np
from PIL import Image, ImageFilter

def extract_edges(image_path):
    # Open the image
    image = Image.open(image_path)

    # Convert the image to grayscale
    grayscale_image = image.convert("L")

    # Apply Sobel edge filter
    edge_image = grayscale_image.filter(ImageFilter.FIND_EDGES)

    return edge_image

def edge_to_bool_array(edge_image, target_width, target_height):
    # Resize the edge image to the target size
    edge_image_resized = edge_image.resize((target_width, target_height))

    # Convert the resized image to a NumPy array
    edge_array = np.array(edge_image_resized)
    return edge_array > 5

def save_bool_array_as_image(bool_array, output_path):
    print(bool_array)
    print(bool_array.shape)
    # Convert bool array to uint8 array
    bool_array_uint8 = (bool_array * 255).astype(np.uint8)

    # Create PIL Image from uint8 array
    bool_image = Image.fromarray(bool_array_uint8, mode='L')

    # Save the image
    bool_image.save(output_path)

def main():
    # Path to the image file
    image_path = "sample.png"

    try:
        # Extract edges
        edge_image = extract_edges(image_path)
        edge_image.save("edge.png")

        # Define target width and height
        target_width, target_height = 64, 90

        # Convert edge image to bool array
        bool_array = edge_to_bool_array(edge_image, target_width, target_height)

        # Save bool array as image
        save_bool_array_as_image(bool_array, "bool_array_image.png")

        print("Bool array image saved as bool_array_image.png")
    except FileNotFoundError:
        print("Image file not found.")

if __name__ == "__main__":
    main()
