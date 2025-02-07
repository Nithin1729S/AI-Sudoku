import random
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

def is_valid(board, row, col, num):
    """Check if placing num at (row, col) is valid in Sudoku."""
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
    start_row, start_col = (row // 3) * 3, (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False
    return True

def solve_sudoku(board):
    """Solve the Sudoku using backtracking."""
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:  # Find an empty spot
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board):  # Recur to solve further
                            return True
                        board[row][col] = 0  # Undo move if it doesn't work
                return False  # No valid number found
    return True  # Puzzle solved

@app.post("/api/recognize")
async def recognize_digits(image: UploadFile = File(..., media_type="image/*")):
    if not image:
        raise HTTPException(status_code=400, detail="No image uploaded")

    try:
        # Read the image for debugging
        content = await image.read()
        print(f"Received image: {image.filename}, Size: {len(content)} bytes")

        # Mocked Sudoku recognition: Generate a random partially filled 9x9 grid
        sudoku_grid = [[random.choice([0, random.randint(1, 9)]) for _ in range(9)] for _ in range(9)]
        print("Sudoku Grid:")
        for row in sudoku_grid:
            print(row)
        return JSONResponse(content={"numbers": sudoku_grid})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/solve")
async def solve_sudoku_api(request: SudokuRequest):
    puzzle = request.puzzle

    # if len(puzzle) != 9 or any(len(row) != 9 for row in puzzle):
    #     raise HTTPException(status_code=400, detail="Invalid Sudoku grid size")

    # if solve_sudoku(puzzle):
    #     return {"solution": puzzle}
    sudoku_grid = [[random.choice([1, random.randint(1, 9)]) for _ in range(9)] for _ in range(9)]
    if (1):
        return {"solution": sudoku_grid}
    else:
        raise HTTPException(status_code=400, detail="No solution exists")

app.include_router(workouts.router)
app.include_router(routines.router)

