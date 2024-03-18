import sys
import os
import random
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree 

def color_diff(a, b):
    return np.sum(np.abs(a - b))

class Cell:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.colors = []
        self.color_weight = []
        self.dominant_index = 0
        self.character_index = 0

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
    
    def get_character(self):
        return self.colors[self.character_index]
    
    def set_KDdominant(self):
        if len(self.colors) < 2:
            return 0
        tree = KDTree(self.colors) 
        dist, ind = tree.query(self.colors, k=2)

        weighted_dist = dist * self.color_weight[:, np.newaxis]
        
        self.dominant_index = np.argmin(weighted_dist[:, 1])
        self.character_index = np.argmax(weighted_dist[:, 1])

class CellImage:
    def __init__(self, image, target_width, target_height=None):
        original_height, original_width = image.shape[:2]

        if target_height is None:
            target_height = int(original_height * (target_width / original_width))

        image = cv2.resize(image, (target_width*4, target_height*4))
        original_height, original_width = image.shape[:2]
        
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
                        color = tuple(image[y, x])
                        color_index = colors.get(color)
                        if color_index is None:
                            colors[color] = len(cell.colors)
                            cell.colors.append(np.array(color))
                            cell.color_weight.append(1)
                        else:
                            cell.color_weight[color_index] += 1

                cell.color_weight = np.array(cell.color_weight)
                cell.colors = np.array(cell.colors)

                self.cells[i][j] = cell

    def get_edge(self, image, threshold=30):
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge_image = cv2.Canny(grayscale_image, threshold, threshold*2)
        edge_image_resized = cv2.resize(edge_image, (self.width, self.height))
        cv2.imwrite("edge.png", edge_image_resized)
        return edge_image_resized

    def set_dominant(self):
        height, width = self.height, self.width
        for i in range(height):
            for j in range(width):
                #print(i, j)
                cell = self.cells[i][j]
                
                '''
                if not cell.rearranged:
                    cell.rearrange()
                
                cell.dominant_index = 0
                '''
                cell.set_KDdominant()

    def extract(self):
        extracted_image = np.empty((len(self.cells), len(self.cells[0]), 3), dtype=np.uint8)
        for i in range(len(self.cells)):
            for j in range(len(self.cells[0])):
                cell = self.cells[i][j]
                if not cell.isOuter or np.sum(cell.get_dominant()) < np.sum(cell.get_character()):
                    extracted_image[i][j] = cell.get_dominant()
                else:
                    t = 0.5
                    extracted_image[i][j] = ((1-t)*cell.get_dominant() + t*cell.get_character().astype('float32')).astype('uint8')
        return extracted_image

def reduce_colors(image, num_colors):
    # Convert the image to RGB mode
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the array to a 2D array of pixels
    pixels = rgb_image.reshape((-1, 3))

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
    new_pixels = new_colors.reshape(rgb_image.shape)

    return cv2.cvtColor(new_pixels.astype(np.uint8), cv2.COLOR_RGB2BGR)

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