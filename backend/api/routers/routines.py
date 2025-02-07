from pydantic import BaseModel
from typing import List, Optional
from fastapi import APIRouter
from sqlalchemy.orm import joinedload
from api.models import Workout, Routine
from api.deps import db_dependency

router = APIRouter(
    prefix='/routines',
    tags=['routines']
)

class RoutineBase(BaseModel):
    name: str
    description: Optional[str] = None

class RoutineCreate(RoutineBase):
    workouts: List[int] = []

@router.get("/")
def get_routines(db: db_dependency):
    # If routines were previously filtered by user, now return all routines or adjust as needed.
    return db.query(Routine).options(joinedload(Routine.workouts)).all()

@router.post("/")
def create_routine(db: db_dependency, routine: RoutineCreate):
    db_routine = Routine(name=routine.name, description=routine.description)
    for workout_id in routine.workouts:
        workout = db.query(Workout).filter(Workout.id == workout_id).first()
        if workout:
            db_routine.workouts.append(workout)
    db.add(db_routine)
    db.commit()
    db.refresh(db_routine)
    return db_routine

@router.delete('/')
def delete_routine(db: db_dependency, routine_id: int):
    db_routine = db.query(Routine).filter(Routine.id == routine_id).first()
    if db_routine:
        db.delete(db_routine)
        db.commit()
    return db_routine
