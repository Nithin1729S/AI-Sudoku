from typing import Annotated
from sqlalchemy.orm import Session
from fastapi import Depends
from .database import SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
# Keep only the database dependency if auth is not needed
db_dependency = Annotated[Session, Depends(get_db)]
