from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from api.views import SudokuRequest, extract_sudoku_grid, solve_sudoku
import cv2 
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

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
        for row in sudoku_grid:
                print(row)
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
