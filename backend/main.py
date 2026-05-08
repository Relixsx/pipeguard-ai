"""
main.py  —  PipeGuard AI v2  —  FastAPI Backend
"""

import asyncio
import csv
import io
import json
from collections import deque
from datetime import datetime
from typing import List, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

# ── Internal imports ──────────────────────────────────────────────────────────
from auth import (
    CreateUserRequest, TokenResponse, UserResponse,
    get_current_user, hash_password, login_for_access_token, require_role,
)
from config import (
    API_DESCRIPTION, API_TITLE, API_VERSION,
    DEFAULT_ANOMALY_PROB, FEATURES, HISTORY_MAXLEN,
    RATE_LIMIT_AUTH, RATE_LIMIT_DEFAULT, RATE_LIMIT_PREDICT, RATE_LIMIT_STREAM,
    SEQ_LEN, STREAM_INTERVAL,
)
from data_simulation import generate_normal_data, get_random_sample, inject_anomalies

# Import DB models with clear names — no shadowing
from database import (
    AnomalyEvent        as DBAnomalyEvent,
    SensorReading       as DBSensorReading,
    User                as DBUser,
    get_db, init_db, seed_admin,
)
from predict import predictor
from preprocessing import df_to_records
from utils import build_event, format_prediction_response, get_logger, stream_payload

logger = get_logger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT_DEFAULT])

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version=API_VERSION)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    lambda req, exc: __import__("fastapi.responses", fromlist=["JSONResponse"])
        .JSONResponse(status_code=429, content={"detail": "Rate limit exceeded."})
)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_anomaly_memory: deque = deque(maxlen=HISTORY_MAXLEN)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event() -> None:
    init_db()
    db = next(get_db())
    seed_admin(db)
    ok = predictor.load()
    if ok:
        logger.info("PipeGuard AI v2 ready | threshold=%.6f", predictor.threshold)
    else:
        logger.warning("Model not loaded — run: python train.py")


# ── Pydantic schemas (named *Input to avoid clashing with DB models) ───────────

class SensorInput(BaseModel):
    pressure:    float = Field(..., example=74.8)
    flow_rate:   float = Field(..., example=4.5)
    temperature: float = Field(..., example=32.1)


class SensorSequenceInput(BaseModel):
    readings: List[SensorInput]


class AcknowledgeRequest(BaseModel):
    notes: Optional[str] = None


# ── DB helper ─────────────────────────────────────────────────────────────────

def _save_reading(db: Session, raw: dict, result: dict, source: str = "stream") -> None:
    try:
        db.add(DBSensorReading(
            timestamp     = datetime.utcnow(),
            pressure      = raw["pressure"],
            flow_rate     = raw["flow_rate"],
            temperature   = raw["temperature"],
            anomaly_score = result.get("anomaly_score", 0.0),
            score_ratio   = result.get("score_ratio",   0.0),
            status        = result.get("status",        "Normal"),
            is_anomaly    = bool(result.get("is_anomaly", False)),
            source        = source,
        ))
        if result.get("is_anomaly"):
            db.add(DBAnomalyEvent(
                timestamp     = datetime.utcnow(),
                pressure      = raw["pressure"],
                flow_rate     = raw["flow_rate"],
                temperature   = raw["temperature"],
                anomaly_score = result.get("anomaly_score", 0.0),
                score_ratio   = result.get("score_ratio",   0.0),
                status        = result.get("status",        "Leak Detected"),
                source        = source,
            ))
            _anomaly_memory.append(build_event(raw, result))
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("DB write error: %s", exc)


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
@limiter.limit(RATE_LIMIT_AUTH)
def login(
    request:   Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db:        Session                   = Depends(get_db),
):
    return login_for_access_token(form_data, db)


@app.get("/auth/me", response_model=UserResponse, tags=["Auth"])
def me(current_user: DBUser = Depends(get_current_user)):
    return current_user


@app.post("/auth/users", response_model=UserResponse, tags=["Auth"])
def create_user(
    payload:      CreateUserRequest,
    db:           Session = Depends(get_db),
    current_user: DBUser  = Depends(require_role("admin")),
):
    from database import User as DBUser2
    if db.query(DBUser2).filter(DBUser2.username == payload.username).first():
        raise HTTPException(status_code=400, detail="Username already exists.")
    user = DBUser2(
        username        = payload.username,
        hashed_password = hash_password(payload.password),
        full_name       = payload.full_name,
        role            = payload.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.get("/auth/users", response_model=List[UserResponse], tags=["Auth"])
def list_users(
    db:           Session = Depends(get_db),
    current_user: DBUser  = Depends(require_role("admin")),
):
    from database import User as DBUser2
    return db.query(DBUser2).all()


# ── System endpoints ──────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health(db: Session = Depends(get_db)):
    total_readings = db.query(DBSensorReading).count()
    total_events   = db.query(DBAnomalyEvent).count()
    unacked        = db.query(DBAnomalyEvent).filter(DBAnomalyEvent.acknowledged == False).count()
    return {
        "status":         "ok",
        "version":        API_VERSION,
        "model_loaded":   predictor.is_ready,
        "threshold":      predictor.threshold,
        "total_readings": total_readings,
        "total_events":   total_events,
        "unacknowledged": unacked,
    }


@app.get("/metrics", tags=["System"])
def metrics(
    db:           Session = Depends(get_db),
    current_user: DBUser  = Depends(get_current_user),
):
    from sqlalchemy import func
    total     = db.query(DBSensorReading).count()
    leaks     = db.query(DBSensorReading).filter(DBSensorReading.is_anomaly == True).count()
    avg_score = db.query(func.avg(DBSensorReading.anomaly_score)).scalar() or 0.0
    return {
        "total_readings":    total,
        "total_anomalies":   leaks,
        "anomaly_rate_pct":  round(100 * leaks / total, 2) if total else 0.0,
        "avg_anomaly_score": round(avg_score, 6),
        "unacknowledged":    db.query(DBAnomalyEvent).filter(DBAnomalyEvent.acknowledged == False).count(),
    }


# ── Simulation endpoints ──────────────────────────────────────────────────────

@app.get("/simulate", tags=["Data"])
@limiter.limit(RATE_LIMIT_DEFAULT)
def simulate(
    request:      Request,
    anomaly:      bool   = Query(False),
    current_user: DBUser = Depends(get_current_user),
):
    return {"reading": get_random_sample(anomaly=anomaly), "forced_anomaly": anomaly}


@app.get("/simulate/sequence", tags=["Data"])
@limiter.limit(RATE_LIMIT_DEFAULT)
def simulate_sequence(
    request:      Request,
    n:            int    = Query(80, ge=10, le=500),
    with_anomaly: bool   = Query(True),
    current_user: DBUser = Depends(get_current_user),
):
    normal_df = generate_normal_data(n_samples=n)
    full_df   = inject_anomalies(normal_df) if with_anomaly else normal_df.copy()
    if not with_anomaly:
        full_df["is_anomaly"] = 0
    return {"data": df_to_records(full_df), "count": len(full_df)}


# ── Prediction endpoints ──────────────────────────────────────────────────────

@app.post("/predict", tags=["Inference"])
@limiter.limit(RATE_LIMIT_PREDICT)
def predict(
    request:      Request,
    payload:      SensorSequenceInput,
    db:           Session = Depends(get_db),
    current_user: DBUser  = Depends(get_current_user),
):
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded. Run: python train.py")
    if len(payload.readings) < SEQ_LEN:
        raise HTTPException(status_code=400, detail=f"Need at least {SEQ_LEN} readings.")
    readings = [r.dict() for r in payload.readings]
    result   = predictor.infer_batch(readings)
    _save_reading(db, readings[-1], result, source="manual")
    return format_prediction_response(result, readings[-1])


@app.post("/predict/single", tags=["Inference"])
@limiter.limit(RATE_LIMIT_PREDICT)
def predict_single(
    request:      Request,
    reading:      SensorInput,
    db:           Session = Depends(get_db),
    current_user: DBUser  = Depends(get_current_user),
):
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    raw    = reading.dict()
    result = predictor.infer_reading(raw)
    _save_reading(db, raw, result, source="manual")
    return format_prediction_response(result, raw)


# ── Event / history endpoints ─────────────────────────────────────────────────

@app.get("/events", tags=["Events"])
def get_events(
    limit:        int    = Query(50, ge=1, le=500),
    unacked_only: bool   = Query(False),
    db:           Session = Depends(get_db),
    current_user: DBUser  = Depends(get_current_user),
):
    q = db.query(DBAnomalyEvent)
    if unacked_only:
        q = q.filter(DBAnomalyEvent.acknowledged == False)
    events = q.order_by(DBAnomalyEvent.timestamp.desc()).limit(limit).all()
    return {
        "events": [e.to_dict() for e in events],
        "total":  db.query(DBAnomalyEvent).count(),
        "unacknowledged": db.query(DBAnomalyEvent).filter(DBAnomalyEvent.acknowledged == False).count(),
    }


@app.post("/events/{event_id}/acknowledge", tags=["Events"])
def acknowledge_event(
    event_id:     int,
    payload:      AcknowledgeRequest,
    db:           Session = Depends(get_db),
    current_user: DBUser  = Depends(get_current_user),
):
    event = db.query(DBAnomalyEvent).filter(DBAnomalyEvent.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found.")
    if event.acknowledged:
        raise HTTPException(status_code=400, detail="Already acknowledged.")
    event.acknowledged    = True
    event.acknowledged_by = current_user.username
    event.acknowledged_at = datetime.utcnow()
    event.notes           = payload.notes
    db.commit()
    return {"ok": True, "event": event.to_dict()}


@app.get("/history", tags=["Events"])
def history(
    limit:        int     = Query(50, ge=1, le=200),
    db:           Session = Depends(get_db),
    current_user: DBUser  = Depends(get_current_user),
):
    events = db.query(DBAnomalyEvent).order_by(DBAnomalyEvent.timestamp.desc()).limit(limit).all()
    return {"events": [e.to_dict() for e in events], "total": db.query(DBAnomalyEvent).count()}


# ── Export endpoints ──────────────────────────────────────────────────────────

@app.get("/export/readings.csv", tags=["Export"])
def export_readings(
    limit:        int     = Query(10000, ge=1, le=100000),
    db:           Session = Depends(get_db),
    current_user: DBUser  = Depends(get_current_user),
):
    rows = db.query(DBSensorReading).order_by(DBSensorReading.timestamp.desc()).limit(limit).all()
    buf  = io.StringIO()
    w    = csv.writer(buf)
    w.writerow(["id","timestamp","pressure","flow_rate","temperature",
                "anomaly_score","score_ratio","status","is_anomaly","source"])
    for r in rows:
        w.writerow([r.id, r.timestamp.isoformat(), r.pressure, r.flow_rate,
                    r.temperature, r.anomaly_score, r.score_ratio,
                    r.status, int(r.is_anomaly), r.source])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=pipeguard_readings.csv"},
    )


@app.get("/export/events.csv", tags=["Export"])
def export_events(
    db:           Session = Depends(get_db),
    current_user: DBUser  = Depends(get_current_user),
):
    events = db.query(DBAnomalyEvent).order_by(DBAnomalyEvent.timestamp.desc()).all()
    buf    = io.StringIO()
    w      = csv.writer(buf)
    w.writerow(["id","timestamp","pressure","flow_rate","temperature",
                "anomaly_score","score_ratio","status","acknowledged",
                "acknowledged_by","acknowledged_at","notes"])
    for e in events:
        w.writerow([e.id, e.timestamp.isoformat(), e.pressure, e.flow_rate,
                    e.temperature, e.anomaly_score, e.score_ratio, e.status,
                    e.acknowledged, e.acknowledged_by or "",
                    e.acknowledged_at.isoformat() if e.acknowledged_at else "",
                    e.notes or ""])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=pipeguard_events.csv"},
    )


# ── SSE Stream ────────────────────────────────────────────────────────────────

@app.get("/stream", tags=["Stream"])
@limiter.limit(RATE_LIMIT_STREAM)
async def stream(
    request:      Request,
    anomaly_prob: float          = Query(DEFAULT_ANOMALY_PROB, ge=0.0, le=1.0),
    token:        Optional[str]  = Query(None),
):
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Resolve username from optional token
    username = "anonymous"
    if token:
        try:
            from jose import jwt as jose_jwt
            from config import SECRET_KEY, ALGORITHM
            payload  = jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub", "anonymous")
        except Exception:
            pass

    async def event_generator():
        rng = np.random.default_rng()
        predictor.reset_buffer()
        db  = next(get_db())
        logger.info("SSE connected | user=%s | anomaly_prob=%.2f", username, anomaly_prob)
        try:
            while True:
                raw    = get_random_sample(anomaly=bool(rng.random() < anomaly_prob))
                result = predictor.infer_reading(raw)
                data   = stream_payload(raw, result, predictor.buffer_size, SEQ_LEN)
                if result.get("is_anomaly"):
                    _save_reading(db, raw, result, source="stream")
                    logger.info(
                        "Stream anomaly | user=%s | score=%.6f | P=%.1f | F=%.1f",
                        username, result["anomaly_score"], raw["pressure"], raw["flow_rate"],
                    )
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(STREAM_INTERVAL)
        except asyncio.CancelledError:
            logger.info("SSE disconnected | user=%s", username)
            db.close()
            raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )
