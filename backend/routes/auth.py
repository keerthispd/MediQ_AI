from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from backend.models.db import get_db
from backend.models.models import User
from backend.services.auth import bearer, create_token, get_current_user, hash_password, revoke_token, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])

# Compared against when the username doesn't exist, so failed logins take the same time either way
_DUMMY_HASH = hash_password("not-a-real-password")


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(max_length=32)
    password: str = Field(max_length=128)


@router.post("/register", status_code=201)
def register(body: RegisterRequest, db=Depends(get_db)):
    user = User(username=body.username, password_hash=hash_password(body.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="That username is already taken.")
    return {"token": create_token(db, user), "username": user.username}


@router.post("/login")
def login(body: LoginRequest, db=Depends(get_db)):
    user = db.query(User).filter(User.username == body.username).first()
    if not verify_password(body.password, user.password_hash if user else _DUMMY_HASH) or user is None:
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    return {"token": create_token(db, user), "username": user.username}


@router.post("/logout")
def logout(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer), db=Depends(get_db)):
    if credentials is not None:
        revoke_token(db, credentials.credentials)
    return {"status": "logged out"}


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return {"username": user.username}
