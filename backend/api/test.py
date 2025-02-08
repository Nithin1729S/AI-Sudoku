import os
from transformers import pipeline
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import cv2
import numpy as np
import re
from typing import List
from pydantic import BaseModel

save_directory = "./model"
model = VisionEncoderDecoderModel.from_pretrained(save_directory)
image_processor = ViTImageProcessor.from_pretrained(save_directory)
tokenizer = AutoTokenizer.from_pretrained(save_directory)

pipe = pipeline(
    "image-to-text",
    model=model,
    feature_extractor=image_processor,
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
            margin = int(cell.shape[0] * 0.1)
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
            misclassification_map = {
                # Common misclassifications due to shape similarity
                'g': '9', 'q': '9', 'b': '6', 'o': '0', 'O': '0', 'D': '0',
                'l': '1', 'I': '1', 'i': '1', '|': '1', '!': '1', 'L': '1',
                's': '5', 'S': '5', 'z': '2', 'Z': '2', 'T': '7',

                # Grid-line artifacts and misclassifications
                '1 1': '1', '0 0': '0', '1|': '1', '|1': '1', 
                '1l': '1', 'l1': '1', 'I1': '1', '1I': '1', '1!': '1', '!1': '1',
                'l|': '1', '|l': '1', 'I|': '1', '|I': '1', 'L|': '1', '|L': '1',

                # Misclassified multi-digit numbers due to grid artifacts
                '11': '1', '12': '2', '13': '3', '14': '4', '15': '5', '16': '6',
                '17': '7', '18': '8', '19': '9', '21': '2', '22': '2', '23': '3',
                '31': '3', '41': '4', '51': '5', '61': '6', '71': '7', '81': '8',
                '91': '9', '2 2': '2', '3 3': '3', '4 4': '4', '5 5': '5',
                '6 6': '6', '7 7': '7', '8 8': '8', '9 9': '9',

                # Misclassifications involving decimal points
                '6.': '6', '7.': '7', '1.': '1', '9.': '9', '0.': '0',
                '.6': '6', '.7': '7', '.1': '1', '.9': '9', '.0': '0',
            }
            # If the recognized text is a single digit, store it; otherwise store 0
            corrected_text = misclassification_map.get(recognized_text, recognized_text)

            # Use a capturing group to get the digit part
            match = re.match(r'^(\d)$', corrected_text)
            if match:
                # Extract the digit from the match and convert to int
                sudoku_grid[row_idx][col_idx] = int(match.group(1))
            else:
                sudoku_grid[row_idx][col_idx] = 0

            

            # Clean up the temporary file
            if os.path.exists(cell_path):
                os.remove(cell_path)

    return sudoku_grid


##############################################
# Backtracking Sudoku Solver Functions
##############################################

class SudokuRequest(BaseModel):
    puzzle: List[List[int]]

def find_empty(board: List[List[int]]):
    """
    Find an empty cell (denoted by 0) in the board.
    Returns a tuple (row, col) or None if no empty cell is found.
    """
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None

def is_valid(board: List[List[int]], row: int, col: int, num: int) -> bool:
    """
    Check if placing 'num' in board[row][col] is valid according to Sudoku rules.
    """
    # Check row and column
    if any(board[row][x] == num for x in range(9)):
        return False
    if any(board[y][col] == num for y in range(9)):
        return False

    # Check the 3x3 sub-grid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == num:
                return False
    return True

def solve_sudoku(board: List[List[int]]) -> bool:
    """
    Solves the Sudoku board in-place using backtracking.
    Returns True if a solution exists, False otherwise.
    """
    empty = find_empty(board)
    if not empty:
        # No empty cell left; solution found
        return True
    row, col = empty

    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve_sudoku(board):
                return True
            board[row][col] = 0  # Backtrack

    return False
