import random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

app.include_router(workouts.router)
app.include_router(routines.router)

