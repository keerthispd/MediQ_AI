"""Password hashing and login tokens, using only the standard library."""
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.models.db import get_db
from backend.models.models import AuthToken, User

TOKEN_TTL = timedelta(days=int(os.getenv("AUTH_TOKEN_DAYS", "7")))

# scrypt parameters: ~16 MB of memory per hash, which makes brute-forcing leaked hashes expensive
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 14, 8, 1

bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        candidate = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p))
    except ValueError:
        return False
    return hmac.compare_digest(candidate.hex(), digest)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def create_token(db, user: User) -> str:
    token = secrets.token_urlsafe(32)
    now = _utcnow()
    db.query(AuthToken).filter(AuthToken.expires_at < now).delete()
    db.add(AuthToken(token_hash=_hash_token(token), user_id=user.id, expires_at=now + TOKEN_TTL))
    db.commit()
    return token


def revoke_token(db, token: str) -> None:
    db.query(AuthToken).filter(AuthToken.token_hash == _hash_token(token)).delete()
    db.commit()


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer), db=Depends(get_db)
) -> User:
    """Dependency that returns the logged-in user or rejects the request with 401."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Please log in.", headers={"WWW-Authenticate": "Bearer"})
    record = db.query(AuthToken).filter(AuthToken.token_hash == _hash_token(credentials.credentials)).first()
    user = db.get(User, record.user_id) if record is not None and record.expires_at > _utcnow() else None
    if user is None:
        raise HTTPException(
            status_code=401, detail="Your session has expired. Please log in again.", headers={"WWW-Authenticate": "Bearer"}
        )
    return user
