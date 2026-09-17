from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class Interaction(Base):
    __tablename__ = 'interactions'
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String, default="default")
    user_message: Mapped[str] = mapped_column(Text)
    assistant_reply: Mapped[Optional[str]] = mapped_column(Text)
    redflag: Mapped[bool] = mapped_column(default=False)
    redflag_details: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), server_default=func.now())


class User(Base):
    __tablename__ = 'users'
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AuthToken(Base):
    __tablename__ = 'auth_tokens'
    id: Mapped[int] = mapped_column(primary_key=True)
    # Only a hash is stored, so a leaked database doesn't expose usable tokens
    token_hash: Mapped[str] = mapped_column(String, unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'))
    expires_at: Mapped[datetime] = mapped_column(DateTime)  # naive UTC
