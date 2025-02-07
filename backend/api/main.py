from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import workouts, routines  # auth router removed

from .database import Base, engine

app = FastAPI()

Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'],  # Adjust as needed
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get("/")
def health_check():
    return {"status": "OK"}

# New function/endpoint to communicate with the frontend
@app.get("/frontend-message")
def frontend_message():
    return {"message": "Hello from the backend!"}

app.include_router(workouts.router)
app.include_router(routines.router)
