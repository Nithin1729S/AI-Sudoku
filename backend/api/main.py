import random
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from api.views import extract_sudoku_grid
import cv2
from .routers import workouts, routines  
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from .database import Base, engine

app = FastAPI()

Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'],  
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get("/")
def health_check():
    return {"status": "OK"}

@app.get("/frontend-message")
def frontend_message():
    return {"message": "Hello from the backend!"}


class SudokuRequest(BaseModel):
    puzzle: List[List[int]]


##############################################
# Backtracking Sudoku Solver Functions
##############################################
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



@app.post("/api/recognize")
async def recognize_digits(image: UploadFile = File(..., media_type="image/*")):
    if not image:
        raise HTTPException(status_code=400, detail="No image uploaded")
    try:
        # Read the image bytes from the uploaded file
        contents = await image.read()
        # Convert bytes data to a NumPy array and decode it (as grayscale)
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # Call the sudoku extraction function
        sudoku_grid = extract_sudoku_grid(img)
        return JSONResponse(content={"numbers": sudoku_grid})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/api/solve")
async def solve_sudoku_api(request: SudokuRequest):
    puzzle = request.puzzle

    # Basic check for grid size
    if len(puzzle) != 9 or any(len(row) != 9 for row in puzzle):
        raise HTTPException(status_code=400, detail="Invalid Sudoku grid size")

    # Attempt to solve the puzzle in place
    if solve_sudoku(puzzle):
        return {"solution": puzzle}
    else:
        raise HTTPException(status_code=400, detail="No solution exists")

app.include_router(workouts.router)
app.include_router(routines.router)

