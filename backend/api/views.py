import os
from transformers import pipeline
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image

save_directory = "./model"
model = VisionEncoderDecoderModel.from_pretrained(save_directory)
feature_extractor = ViTFeatureExtractor.from_pretrained(save_directory)
tokenizer = AutoTokenizer.from_pretrained(save_directory)

pipe = pipeline(
    "image-to-text",
    model=model,
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    framework="pt"
)

def extract_sudoku_grid(img: np.ndarray) -> list:
    """
    Given an image (as a NumPy array), this function locates the Sudoku grid,
    performs a perspective transform, splits the grid into cells, checks for blank
    cells using a standard deviation threshold, and uses the OCR pipeline to
    recognize digits. It returns a 9x9 matrix (list of lists) representing the Sudoku.
    """
    # Ensure the image is in grayscale
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    # Pre-process the image: blur, threshold, invert, and dilate
    proc = cv2.GaussianBlur(img_gray.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(
        proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    proc = cv2.bitwise_not(proc, proc)

    kernel = np.array([[0., 1., 0.],
                       [1., 1., 1.],
                       [0., 1., 0.]], np.uint8)
    proc = cv2.dilate(proc, kernel)

    # Find contours in the image
    contours, _ = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise Exception("No contours found in the image.")

    # Sort contours by area (largest first) and assume the largest is the sudoku grid
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]

    # Helper function to compute Euclidean distance between two points
    def distance_between(p1, p2):
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    # Find the four extreme points of the contour
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=lambda x: x[1])
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=lambda x: x[1])
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=lambda x: x[1])
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=lambda x: x[1])

    # Extract the four corners in order: top-left, top-right, bottom-right, bottom-left
    crop_rect = [
        polygon[top_left][0],
        polygon[top_right][0],
        polygon[bottom_right][0],
        polygon[bottom_left][0]
    ]

    # Compute the side length of the square
    side = max([
        distance_between(crop_rect[0], crop_rect[1]),
        distance_between(crop_rect[1], crop_rect[2]),
        distance_between(crop_rect[2], crop_rect[3]),
        distance_between(crop_rect[3], crop_rect[0])
    ])
    side = int(side)

    # Define destination points and perform the perspective transform
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(np.array(crop_rect, dtype='float32'), dst)
    warped = cv2.warpPerspective(img_gray, M, (side, side))

    # Divide the warped image into a 9x9 grid of cells
    cell_side = warped.shape[0] // 9
    squares = []
    for j in range(9):
        row = []
        for i in range(9):
            p1 = (i * cell_side, j * cell_side)
            p2 = ((i + 1) * cell_side, (j + 1) * cell_side)
            row.append((p1, p2))
        squares.append(row)

    # Prepare an empty 9x9 Sudoku grid
    sudoku_grid = [[0 for _ in range(9)] for _ in range(9)]
    blank_threshold = 10  # Standard deviation threshold to detect blank cells

    # Process each cell
    for row_idx, row in enumerate(squares):
        for col_idx, square in enumerate(row):
            p1, p2 = square
            # Extract the cell from the warped image
            cell = warped[int(p1[1]):int(p2[1]), int(p1[0]):int(p2[0])]

            # Crop cell borders (15% margin)
            margin = int(cell.shape[0] * 0.15)
            if cell.shape[0] - 2 * margin > 0 and cell.shape[1] - 2 * margin > 0:
                cell = cell[margin:-margin, margin:-margin]

            # Check if the cell is blank based on its standard deviation
            if np.std(cell) < blank_threshold:
                sudoku_grid[row_idx][col_idx] = 0
                continue

            # Save the cell temporarily (the OCR pipeline expects a file path)
            cell_path = f"cell_{row_idx}_{col_idx}.png"
            cv2.imwrite(cell_path, cell)

            # Use the OCR pipeline to recognize text from the cell image
            result = pipe(cell_path)
            recognized_text = result[0]['generated_text'].strip()

            # If the recognized text is a single digit, store it; otherwise store 0
            if re.match(r'^\d$', recognized_text):
                sudoku_grid[row_idx][col_idx] = int(recognized_text)
            else:
                sudoku_grid[row_idx][col_idx] = 0

            # Clean up the temporary file
            if os.path.exists(cell_path):
                os.remove(cell_path)

    return sudoku_grid