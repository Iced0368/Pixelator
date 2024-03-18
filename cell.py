import sys, os
import random
from PIL import Image, ImageFilter
import cv2
import numpy as np
from sklearn.cluster import KMeans

def color_diff(a, b):
    return np.sum(np.abs(a - b))

class Cell:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.colors = []
        self.color_weight = []
        self.dominant_index = 0

        self.rearranged = False
        self.isOuter = False

    def rearrange(self):
        diffs = np.zeros(shape=(len(self.colors), len(self.colors)))

        for i in range(len(self.colors)):
            for j in range(0, i):
                diff = color_diff(self.colors[i], self.colors[j])
                diffs[i, j] = diff * self.color_weight[j]
                diffs[j, i] = diff * self.color_weight[i]

        sums = np.sum(diffs, axis=1)
        sorted_indices = np.argsort(sums)

        rearranged_colors = [None] * len(self.colors)
        rearranged_weights = [None] * len(self.color_weight)

        for i, index in enumerate(sorted_indices):
            rearranged_colors[i] = self.colors[index]
            rearranged_weights[i] = self.color_weight[index]
            if index == self.dominant_index:
                self.dominant_index = index
        
        self.colors = rearranged_colors
        self.color_weight = rearranged_weights
        self.rearranged = True
        
    def get_weighted_index(self, index):
        index_sum = 0
        for i in range(len(self.colors)):
            index_sum += self.color_weight[i]
            if index_sum >= index:
                return i
        return -1
    
    def get_dominant(self):
        return self.colors[self.dominant_index]

class CellImage:
    def __init__(self, image, target_width, target_height=None):
        original_width, original_height = image.size

        if target_height is None:
            target_height = int(original_height * (target_width / original_width))

        image = image.resize((target_width*4, target_height*4))
        original_width, original_height = image.size
        
        cell_width = original_width // target_width
        cell_height = original_height // target_height

        self.width, self.height = target_width, target_height
        self.cells = np.empty((target_height, target_width), dtype=Cell)

        edge = self.get_edge(image)

        for i in range(target_height):
            for j in range(target_width):
                cell = Cell(cell_width, cell_height)
                cell.isOuter = edge[i][j]
                colors = {}
                for y in range(i * cell_height, (i + 1) * cell_height):
                    for x in range(j * cell_width, (j + 1) * cell_width):
                        color = image.getpixel((x, y))
                        color_index = colors.get(color)
                        if color_index is None:
                            colors[color] = len(cell.colors)
                            cell.colors.append(np.array(color))
                            cell.color_weight.append(1)
                        else:
                            cell.color_weight[color_index] += 1

                self.cells[i][j] = cell

    def get_edge(self, image, threshold=50):
        grayscale_image = image.convert("L")
        edge_image = grayscale_image.filter(ImageFilter.FIND_EDGES)
        edge_image_resized = edge_image.resize((self.width, self.height))
        edge_array = np.array(edge_image_resized)
        result = edge_array > threshold

        bool_array_uint8 = (result * 255).astype(np.uint8)
        # Create PIL Image from uint8 array
        bool_image = Image.fromarray(bool_array_uint8, mode='L')

        # Save the image
        bool_image.save('edge.png')
        return result


    def set_dominant(self):
        height, width = self.height, self.width
        for i in range(height):
            for j in range(width):
                print(i, j)
                cell = self.cells[i][j]
                if not cell.rearranged:
                    cell.rearrange()
                cell.dominant_index = 0



    def extract(self):
        extracted_image = np.empty((len(self.cells), len(self.cells[0]), 3), dtype=np.uint8)
        for i in range(len(self.cells)):
            for j in range(len(self.cells[0])):
                cell = self.cells[i][j]
                if not cell.isOuter or np.sum(cell.get_dominant()) < np.sum(cell.colors[-1]):
                    extracted_image[i][j] = cell.get_dominant()
                else:
                    t = 0.5#cell.color_weight[-1] / (cell.height * cell.width)
                    extracted_image[i][j] = ((1-t)*cell.get_dominant() + t*cell.colors[-1].astype('float32')).astype('uint8')
        return extracted_image

def reduce_colors(image, num_colors):
    # Convert the image to RGB mode
    rgb_image = image.convert("RGB")

    # Convert the PIL image to a NumPy array
    numpy_array = np.array(rgb_image)

    # Reshape the array to a 2D array of pixels
    pixels = numpy_array.reshape((-1, 3))

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_colors, n_init=num_colors*4)
    kmeans.fit(pixels)

    # Find the most common color in each cluster
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    most_common_labels = unique[np.argsort(counts)][::-1][:num_colors]

    # Replace each pixel value with its nearest centroid value
    new_colors = kmeans.cluster_centers_[most_common_labels]
    labels_map = {k: v for k, v in zip(unique, kmeans.cluster_centers_)}
    new_colors = np.array([labels_map[label] for label in kmeans.labels_])

    # Reshape the array back to the original image shape
    new_pixels = new_colors.reshape(numpy_array.shape)

    # Convert NumPy array back to PIL image
    new_image = Image.fromarray(new_pixels.astype(np.uint8), "RGB")

    return new_image

def calculate_discontinuity(cell_image):
    height, width = cell_image.height, cell_image.width
    diffs = []

    for i in range(height):
        for j in range(width):
            cell = cell_image.cells[i][j]
            min_diff = float('inf')
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if 0 <= i + di < height and 0 <= j + dj < width and not (di == 0 and dj == 0):
                        neighbor_cell = cell_image.cells[i + di][j + dj]
                        diff = color_diff(cell.get_dominant(), neighbor_cell.get_dominant())
                        min_diff = min(min_diff, diff)
            diffs.append(min_diff)

    return np.mean(diffs)