"""Same-origin public research demo. No credentials or training at startup."""
import json
import os
import sys
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Existing Render services may still launch `cd backend && uvicorn main:app`.
# Resolve our package from the repository root in that entrypoint as well.
if not __package__:
    sys.path.insert(0, str(ROOT))
from backend.experiments import SCENARIOS, RunStore, create_analysis, create_simulation, run_csv
from backend.runtime import Runtime
from backend.simulator import FEATURES
from backend.paper_labs import network_lab, information_lab, acoustic_lab, analyze_acoustic

MAX_BODY_BYTES = 512_000


class ScenarioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scenario: Literal["healthy", "leak", "gradual_leak", "normal_transient", "sensor_bias", "missing_sensor", "weak_excitation"] = "leak"
    leak_fraction: Literal[0.02, 0.05, 0.1] = 0.05
    location: int = Field(default=1, ge=0, le=2)
    regime: Literal["in_range", "shifted"] = "shifted"
    seed: int = Field(default=82001, ge=0, le=2_147_483_647)


class Reading(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    timestamp_s: float
    values: list[float | None] = Field(min_length=8, max_length=8)

    @field_validator("values")
    @classmethod
    def physical_units(cls, values):
        # Reject impossible units, but preserve explicit missing sensors.
        for i in (0, 1, 2, 6):
            if values[i] is not None and not 10_000 <= values[i] <= 20_000_000:
                raise ValueError("Pressures must be absolute Pa between 10 kPa and 20 MPa")
        if values[5] is not None and not 150 <= values[5] <= 500:
            raise ValueError("Temperature must be Kelvin between 150 and 500")
        for i in (3, 4):
            if values[i] is not None and not -100 <= values[i] <= 100:
                raise ValueError("Mass flows must be kg/s between -100 and 100")
        if values[7] is not None and not 0 <= values[7] <= 2:
            raise ValueError("Valve command must be dimensionless between 0 and 2")
        return values


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    readings: list[Reading] = Field(min_length=32, max_length=1200)


class NetworkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    seed: int = Field(default=91001, ge=0, le=2_147_483_640)
    scenario: Literal["confounded", "independent_noise", "weak_excitation", "target_change", "control_change"] = "confounded"
    sensor_mode: Literal["selected", "target_only"] = "selected"


class InformationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    excitation: Literal["target_and_separator", "wrong_location"] = "target_and_separator"


class AcousticRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    seed: int = Field(default=92001, ge=0, le=2_147_483_640)
    scenario: Literal["independent", "source_change", "correlated", "colored", "dependent_uncorrelated"] = "source_change"


class AcousticUpload(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    measurements: list[list[float]] = Field(min_length=768, max_length=4096)
    healthy_samples: int = Field(default=1024, ge=512, le=3840)
    order: int = Field(default=4, ge=1, le=12)
    radius: int = Field(default=1, ge=1, le=2)
    adjacent_delay: int = Field(default=3, ge=1, le=8)

    @field_validator("measurements")
    @classmethod
    def synchronized(cls, rows):
        channels = len(rows[0])
        if not 2 <= channels <= 8 or any(len(row) != channels for row in rows):
            raise ValueError("Every row must contain the same 2-8 synchronized finite channels")
        if any(abs(v) > 1e12 for row in rows for v in row):
            raise ValueError("Channel magnitude exceeds the numeric range")
        return rows


class RateLimit:
    """Bounded per-client compute allowance, not a persistent identity system."""
    def __init__(self):
        self.clients = defaultdict(deque)
        self.lock = threading.Lock()

    def accept(self, client):
        with self.lock:
            now = time.monotonic()
            # Evict expired addresses as well as old timestamps.
            for key in list(self.clients):
                queue = self.clients[key]
                while queue and now - queue[0] > 600:
                    queue.popleft()
                if not queue:
                    del self.clients[key]
            if len(self.clients) >= 1000 and client not in self.clients:
                return False
            queue = self.clients[client]
            if len(queue) >= 20:
                return False
            queue.append(now)
            return True


def create_app():
    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = Runtime(ROOT / "deployment/models")
        app.state.runs = RunStore()
        app.state.compute = threading.BoundedSemaphore(1)
        app.state.limiter = RateLimit()
        yield

    app = FastAPI(title="PipeGuard Research Lab", version="3.0.0", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def request_bounds(request, call_next):
        # Check the actual stream length, including requests without Content-Length.
        if request.method == "POST":
            chunks, length = [], 0
            async for chunk in request.stream():
                length += len(chunk)
                if length > MAX_BODY_BYTES:
                    return Response("Request exceeds 512 kB", status_code=413)
                chunks.append(chunk)
            request._body = b"".join(chunks)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        script_policy = "'self'"
        style_policy = "'self' 'unsafe-inline'"
        # Only optional Swagger docs use a CDN, with a nonced initializer.
        if request.url.path == "/docs" and hasattr(request.state, "docs_nonce"):
            script_policy += f" https://cdn.jsdelivr.net 'nonce-{request.state.docs_nonce}'"
            style_policy += " https://cdn.jsdelivr.net"
        response.headers["Content-Security-Policy"] = f"default-src 'self'; script-src {script_policy}; style-src {style_policy}; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api/runs") else "no-cache"
        return response

    @app.get("/docs", include_in_schema=False)
    def api_docs(request: Request):
        import secrets
        from fastapi.openapi.docs import get_swagger_ui_html
        from fastapi.responses import HTMLResponse
        request.state.docs_nonce = secrets.token_urlsafe(18)
        html = get_swagger_ui_html(openapi_url="/openapi.json", title="PipeGuard · API", swagger_favicon_url="/assets/icon.svg")
        body = html.body.decode().replace("<script>", f'<script nonce="{request.state.docs_nonce}">')
        return HTMLResponse(body)

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(ROOT / "frontend/index.html")

    @app.get("/login.html", include_in_schema=False)
    def former_login():
        from fastapi.responses import RedirectResponse
        return RedirectResponse("/", status_code=307)

    @app.get("/health")
    def health(request: Request):
        ready = hasattr(request.app.state, "runtime")
        if not ready:
            raise HTTPException(503, "Models are not ready")
        return {"status": "ready", "version": "3.0.0", "models_loaded": 5,
                "neural_runtime": "ONNX CPU", "scope": "simulation research demo"}

    @app.get("/api/config")
    def config(request: Request):
        return {"models": request.app.state.runtime.config, "scenarios": SCENARIOS,
                "max_readings": 1200, "run_expiry_s": 3600}

    def compute(request, factory, payload):
        client = request.client.host if request.client else "unknown"
        if not request.app.state.limiter.accept(client):
            raise HTTPException(429, "Compute limit reached. Try again in ten minutes.", headers={"Retry-After": "600"})
        semaphore = request.app.state.compute
        if not semaphore.acquire(blocking=False):
            raise HTTPException(503, "Another experiment is running. Try again shortly.", headers={"Retry-After": "5"})
        try:
            started = time.perf_counter()
            result = factory(request.app.state.runtime, payload)
            result["compute_seconds"] = round(time.perf_counter() - started, 3)
            return request.app.state.runs.add(result)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        finally:
            semaphore.release()

    @app.post("/api/runs", status_code=201)
    def new_run(payload: ScenarioRequest, request: Request):
        return compute(request, create_simulation, payload)

    @app.post("/api/analyze", status_code=201)
    def analyze(payload: AnalysisRequest, request: Request):
        return compute(request, create_analysis, payload)

    @app.post("/api/labs/network", status_code=201)
    def network(payload: NetworkRequest, request: Request):
        return compute(request, lambda _, p: network_lab(**p.model_dump()), payload)

    @app.post("/api/labs/information", status_code=201)
    def information(payload: InformationRequest, request: Request):
        return compute(request, lambda _, p: information_lab(**p.model_dump()), payload)

    @app.post("/api/labs/acoustic", status_code=201)
    def acoustic(payload: AcousticRequest, request: Request):
        return compute(request, lambda _, p: acoustic_lab(**p.model_dump()), payload)

    @app.post("/api/labs/acoustic/analyze", status_code=201)
    def acoustic_upload(payload: AcousticUpload, request: Request):
        if len(payload.measurements)-payload.healthy_samples < 256:
            raise HTTPException(422, "Reserve at least 256 subsequent samples for monitoring")
        return compute(request, lambda _, p: analyze_acoustic(**p.model_dump()), payload)

    @app.get("/api/paper-methods")
    def paper_methods():
        root = ROOT / "deployment/evidence"
        return {key: json.loads((root / filename).read_text()) for key, filename in {
            "examples": "paper_methods_examples.json", "benchmarks": "paper_methods_results.json",
            "context": "context_refinement_results.json", "ablation": "context_ablation_results.json",
            "recorded_transfer": "recorded_source_transfer.json"}.items()}

    def get_run(request, run_id):
        if run_id == "example":
            return json.loads((ROOT / "deployment/evidence/example.json").read_text())
        run = request.app.state.runs.get(run_id)
        if run is None:
            raise HTTPException(404, "Run expired or not found. Generate it again with the same seed.")
        return run

    @app.get("/api/example")
    def example():
        return FileResponse(ROOT / "deployment/evidence/example.json", media_type="application/json")

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str, request: Request):
        return get_run(request, run_id)

    @app.get("/api/runs/{run_id}/export.csv")
    def csv_export(run_id: str, request: Request):
        run = get_run(request, run_id)
        if "predictions" not in run:
            raise HTTPException(422, "Paper-method labs export their full results as JSON")
        content = run_csv(run)
        return Response(content, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="pipeguard-{run_id}.csv"'})

    @app.get("/api/runs/{run_id}/export.json")
    def json_export(run_id: str, request: Request):
        content = json.dumps(get_run(request, run_id), allow_nan=False, indent=2)
        return Response(content, media_type="application/json", headers={"Content-Disposition": f'attachment; filename="pipeguard-{run_id}.json"'})

    @app.get("/api/evidence")
    def evidence():
        root = ROOT / "deployment/evidence"
        final = json.loads((root / "regime_results.json").read_text())
        acoustic = json.loads((root / "acoustic_results.json").read_text())
        return {
            "simulation": {"summaries": final["summaries"], "protocol": final["protocol"],
                           "event_definition": final["event_definition"], "test_seed_range": final["test_seed_range"]},
            "acoustic": {"summaries": acoustic["summaries"], "audit": acoustic["audit"], "scope": acoustic["scope"]},
            "references": json.loads((root / "publication_references.json").read_text()),
            "datasets": json.loads((root / "dataset_qualification.json").read_text()),
            "manifest": json.loads((ROOT / "deployment/models/manifest.json").read_text()),
            "runtime_validation": json.loads((root / "validation.json").read_text()),
            "upgrade_validation": json.loads((root / "upgrade-validation.json").read_text()),
        }

    files = {
        "report": "research-report.pdf", "results": "regime_results.json",
        "primary-results": "benchmark_results.json", "acoustic-results": "acoustic_results.json",
        "paper-mapping": "PAPER_MAPPING.md", "data-access": "DATA_ACCESS.md",
        "reading-ledger": "paper_reading_ledger.json", "source-audit": "source_audit.json",
        "validation": "validation.json",
        "method-audit": "PAPER_METHODS_AUDIT.md", "method-results": "paper_methods_results.json",
        "context-results": "context_refinement_results.json", "context-ablation": "context_ablation_results.json",
        "recorded-transfer": "recorded_source_transfer.json",
        "upgrade-validation": "upgrade-validation.json",
    }

    @app.get("/api/download/{name}")
    def download(name: str):
        if name not in files:
            raise HTTPException(404, "Unknown evidence file")
        return FileResponse(ROOT / "deployment/evidence" / files[name], filename=files[name])

    @app.get("/api/template.csv")
    def template():
        from backend.simulator import simulate
        run = simulate("template", 82002, "healthy")
        import csv
        import io
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["timestamp_s", *FEATURES])
        writer.writerows([[int(t), *map(float, x)] for t, x in zip(run["time_s"][:120], run["x"][:120])])
        return Response(out.getvalue(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="pipeguard-sensor-template.csv"'})

    app.mount("/assets", StaticFiles(directory=ROOT / "frontend"), name="assets")
    return app


app = create_app()
