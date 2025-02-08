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
    Given an image (as a NumPy array), this function finds the Sudoku puzzle,
    performs a perspective transform, splits it into cells, and uses the OCR pipeline
    to recognize the digits. It returns a 9x9 grid (list of lists) of integers.
    """
    # If the image is not already grayscale, convert it
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Pre-process the image to highlight the grid
    proc = cv2.GaussianBlur(gray.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(
        proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    proc = cv2.bitwise_not(proc, proc)

    # Dilate the image to enhance grid lines
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    proc = cv2.dilate(proc, kernel)

    # Find contours in the image
    contours, _ = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise Exception("No contours found, cannot locate sudoku grid.")

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]

    # Helper function: Euclidean distance between two points
    def distance_between(p1, p2):
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    # Find the four extreme points of the contour
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=lambda x: x[1])
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=lambda x: x[1])
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=lambda x: x[1])
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=lambda x: x[1])

    # Arrange the corners in order: top-left, top-right, bottom-right, bottom-left
    crop_rect = [
        polygon[top_left][0],
        polygon[top_right][0],
        polygon[bottom_right][0],
        polygon[bottom_left][0]
    ]

    # Compute the maximum side length of the detected square
    side = max([
        distance_between(crop_rect[0], crop_rect[1]),
        distance_between(crop_rect[1], crop_rect[2]),
        distance_between(crop_rect[2], crop_rect[3]),
        distance_between(crop_rect[3], crop_rect[0])
    ])
    #side = int(side)

    # Set destination points for the perspective transform (a square)
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(np.array(crop_rect, dtype="float32"), dst)
    warped = cv2.warpPerspective(gray, M, (side, side))

    # Divide the warped image into a 9x9 grid of cells
    cell_side = warped.shape[0] // 9
    squares = []
    for j in range(9):
        row = []
        for i in range(9):
            p1 = (i * cell_side, j * cell_side)      # Top-left corner of the cell
            p2 = ((i + 1) * cell_side, (j + 1) * cell_side)  # Bottom-right corner of the cell
            row.append((p1, p2))
        squares.append(row)

    # Create a 9x9 grid to store recognized numbers
    sudoku_grid = [[0 for _ in range(9)] for _ in range(9)]

    # Process each cell with the OCR pipeline
    for row_idx, row in enumerate(squares):
        for col_idx, square in enumerate(row):
            (x1, y1), (x2, y2) = square
            cell = warped[y1:y2, x1:x2]

            # Optionally crop out borders (adjust the margin percentage as needed)
            margin = int(cell.shape[0] * 0.15)
            if margin * 2 < cell.shape[0]:
                cropped_cell = cell[margin:-margin, margin:-margin]
            else:
                cropped_cell = cell

            # Convert the cell image to RGB (the pipeline may expect 3 channels)
            cell_rgb = cv2.cvtColor(cropped_cell, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(cell_rgb)

            # Use the pipeline to recognize text from the cell image
            result = pipe(pil_img)
            recognized_text = result[0]['generated_text'].strip()

            # If a single digit is recognized, update the grid; otherwise, leave it as 0.
            if re.match(r'^\d$', recognized_text):
                sudoku_grid[row_idx][col_idx] = int(recognized_text)
            else:
                sudoku_grid[row_idx][col_idx] = 0

    return sudoku_grid