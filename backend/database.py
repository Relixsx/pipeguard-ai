"""
database.py
-----------
SQLAlchemy database models and session management.

Tables:
  users           — operator accounts with roles
  sensor_readings — every reading processed (stream, manual, batch)
  anomaly_events  — flagged anomalies with acknowledgement workflow

SQLite by default; swap to PostgreSQL by setting DATABASE_URL in .env
"""

from datetime import datetime
from typing import Generator

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Index,
    Integer, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import DATABASE_URL
from utils  import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Engine & Session
# ─────────────────────────────────────────────────────────────────────────────

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    echo=False,           # set True to log all SQL (very verbose)
    pool_pre_ping=True,   # verify connection health before using it
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ─────────────────────────────────────────────────────────────────────────────
#  Base
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
#  Models
# ─────────────────────────────────────────────────────────────────────────────

class User(Base):
    """Operator / admin account."""
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String(64),  unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    full_name       = Column(String(128), nullable=True)
    role            = Column(String(32),  default="operator")   # admin | operator | viewer
    is_active       = Column(Boolean,     default=True)
    created_at      = Column(DateTime,    default=datetime.utcnow)
    last_login      = Column(DateTime,    nullable=True)

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"


class SensorReading(Base):
    """
    Every sensor reading processed by the system.
    Persisted for audit trail, export, and retrospective analysis.
    """
    __tablename__ = "sensor_readings"

    id            = Column(Integer,  primary_key=True, index=True)
    timestamp     = Column(DateTime, default=datetime.utcnow, nullable=False)
    pressure      = Column(Float,    nullable=False)
    flow_rate     = Column(Float,    nullable=False)
    temperature   = Column(Float,    nullable=False)
    anomaly_score = Column(Float,    default=0.0)
    score_ratio   = Column(Float,    default=0.0)
    status        = Column(String(32), default="Normal")
    is_anomaly    = Column(Boolean,  default=False)
    source        = Column(String(32), default="stream")  # stream | manual | batch

    __table_args__ = (
        Index("ix_readings_timestamp", "timestamp"),
        Index("ix_readings_is_anomaly", "is_anomaly"),
    )


class AnomalyEvent(Base):
    """
    Detected anomaly events with full acknowledgement workflow.
    Each event represents a Leak Detected or Warning classification.
    """
    __tablename__ = "anomaly_events"

    id                = Column(Integer,  primary_key=True, index=True)
    timestamp         = Column(DateTime, default=datetime.utcnow, nullable=False)
    pressure          = Column(Float,    nullable=False)
    flow_rate         = Column(Float,    nullable=False)
    temperature       = Column(Float,    nullable=False)
    anomaly_score     = Column(Float,    nullable=False)
    score_ratio       = Column(Float,    nullable=False)
    status            = Column(String(32), nullable=False)       # Warning | Leak Detected
    source            = Column(String(32), default="stream")

    # Acknowledgement
    acknowledged      = Column(Boolean,     default=False)
    acknowledged_by   = Column(String(64),  nullable=True)
    acknowledged_at   = Column(DateTime,    nullable=True)
    notes             = Column(Text,        nullable=True)

    __table_args__ = (
        Index("ix_events_timestamp",    "timestamp"),
        Index("ix_events_acknowledged", "acknowledged"),
    )

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "timestamp":        self.timestamp.isoformat() + "Z",
            "pressure":         self.pressure,
            "flow_rate":        self.flow_rate,
            "temperature":      self.temperature,
            "anomaly_score":    self.anomaly_score,
            "score_ratio":      self.score_ratio,
            "status":           self.status,
            "source":           self.source,
            "acknowledged":     self.acknowledged,
            "acknowledged_by":  self.acknowledged_by,
            "acknowledged_at":  self.acknowledged_at.isoformat() + "Z" if self.acknowledged_at else None,
            "notes":            self.notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Session Dependency  (used by FastAPI Depends)
# ─────────────────────────────────────────────────────────────────────────────

def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised → %s", DATABASE_URL)


def seed_admin(db: Session) -> None:
    """
    Create the default admin account on first startup if no users exist.
    Credentials are set via ADMIN_USERNAME / ADMIN_PASSWORD env vars.
    """
    from config import DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD, DEFAULT_ADMIN_FULLNAME
    from auth import hash_password

    if db.query(User).count() == 0:
        admin = User(
            username        = DEFAULT_ADMIN_USERNAME,
            hashed_password = hash_password(DEFAULT_ADMIN_PASSWORD),
            full_name       = DEFAULT_ADMIN_FULLNAME,
            role            = "admin",
            is_active       = True,
        )
        db.add(admin)
        db.commit()
        logger.info(
            "Default admin created | username: %s | password: %s",
            DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD,
        )
