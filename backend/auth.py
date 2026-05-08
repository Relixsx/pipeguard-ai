"""
auth.py
-------
JWT authentication and authorisation for PipeGuard AI.

Provides:
  - Password hashing  (bcrypt via passlib)
  - JWT creation / verification  (python-jose)
  - FastAPI dependencies: get_current_user, require_role
  - /auth/login and /auth/me endpoint handlers (called from main.py)
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import (
    SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES,
)
from database import User, get_db
from utils import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Password Hashing
# ─────────────────────────────────────────────────────────────────────────────

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Hash a plain-text password using bcrypt."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plain-text password against a bcrypt hash."""
    return _pwd_context.verify(plain, hashed)


# ─────────────────────────────────────────────────────────────────────────────
#  JWT
# ─────────────────────────────────────────────────────────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    expire    = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# ─────────────────────────────────────────────────────────────────────────────
#  FastAPI Dependencies
# ─────────────────────────────────────────────────────────────────────────────

def get_current_user(
    token: str         = Depends(oauth2_scheme),
    db:    Session     = Depends(get_db),
) -> User:
    """
    FastAPI dependency — decode JWT and return the authenticated User.
    Raises 401 if token is invalid, expired, or user is deactivated.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token. Please log in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username : str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None or not user.is_active:
        raise credentials_exception

    return user


def require_role(*roles: str):
    """
    Factory for role-based access control.

    Usage:
        @app.delete("/users/{id}")
        def delete_user(current_user = Depends(require_role("admin"))):
            ...
    """
    def _checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {' or '.join(roles)}",
            )
        return current_user
    return _checker


# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    username:     str
    full_name:    Optional[str]
    role:         str


class UserResponse(BaseModel):
    id:         int
    username:   str
    full_name:  Optional[str]
    role:       str
    is_active:  bool
    created_at: datetime
    last_login: Optional[datetime]

    class Config:
        from_attributes = True


class CreateUserRequest(BaseModel):
    username:  str
    password:  str
    full_name: Optional[str] = None
    role:      str = "operator"


# ─────────────────────────────────────────────────────────────────────────────
#  Auth Logic  (called from main.py endpoint handlers)
# ─────────────────────────────────────────────────────────────────────────────

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Return User if credentials are valid, else None."""
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def login_for_access_token(
    form_data: OAuth2PasswordRequestForm,
    db:        Session,
) -> TokenResponse:
    """
    Validate credentials, update last_login, return JWT token.
    Raises 401 on invalid credentials.
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        logger.warning("Failed login attempt for username: %s", form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled. Contact your administrator.",
        )

    # Update last_login timestamp
    user.last_login = datetime.utcnow()
    db.commit()

    token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    logger.info("User logged in: %s (%s)", user.username, user.role)

    return TokenResponse(
        access_token=token,
        username=user.username,
        full_name=user.full_name,
        role=user.role,
    )
