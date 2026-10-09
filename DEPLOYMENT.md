# Deploy the actual PipeGuard application

The repository contains the frontend, inference API, frozen weights and evidence files. Serve them as **one web service**. A static HTML-only host cannot run these models.

## Update an existing Render Python service

After the reviewed code is in the GitHub repository:

1. In the existing service settings, choose the branch containing this update. For review, use `feat/research-demo`; use `main` after merging the reviewed update.
2. Set **Root Directory** to blank (repository root), **Build Command** to `sh build.sh`, **Start Command** to `sh start.sh`, and **Health Check Path** to `/health`.
3. Use Python 3.12; `.python-version` supplies it. The build verifies frozen artifacts and never trains. For existing services, the previous start command `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT` remains supported and has been tested with real inference; `sh start.sh` is the preferred command for new services.
4. If automatic deploys are enabled, merging into the watched branch starts the deploy. Otherwise trigger a deploy. Wait for `/health` to return `status: ready` and `models_loaded: 4`.
5. Open the public URL. The landing page should say **PipeGuard Research Lab**, show an actual scored example, and allow a new experiment without login.

The browser uses relative API URLs, so the public address can change without editing frontend code. No JWT secret, default admin credentials, database, GPU or persistent disk is required for this demo.

## New Render Docker service

`render.yaml` is a blueprint for a **new** free Docker demo service. It does not silently migrate or replace an existing service. Connect the updated repository/branch in Render, review the blueprint, and deploy. The blueprint selects the free plan explicitly and disables automatic deploy triggers.

For manual Docker setup, choose the repository root, Dockerfile `./Dockerfile`, health check `/health`, and set `PORT=10000`. `start.sh` binds the assigned port. Render terminates HTTPS for the browser.

Official references, checked 9 October 2026:
- [Blueprint specification](https://render.com/docs/blueprint-spec)
- [Web services and ports](https://render.com/docs/web-services)
- [HTTP readiness checks](https://render.com/docs/health-checks)
- [Free-service limitations](https://render.com/docs/free)
- [Compute plans](https://render.com/docs/compute-plans)

Render's free service may sleep after inactivity and has limited CPU and memory. Cold starts can delay the first visit. The measured workspace compute/memory figures are not performance guarantees for a throttled host. Open the service before the interview, wait for readiness, and export important runs. No paid upgrade is necessary to configure this blueprint.

## Docker locally / another container host

```sh
docker build -t pipeguard-research .
docker run --rm -p 8000:8000 -e PORT=8000 pipeguard-research
```

The container runs as a non-root user, serves both UI and API, and includes its HTTP readiness check. It excludes the legacy app, original data, training dependencies and PyTorch checkpoints. The two exported neural models total about 1.16 MB.

`railway.json` also specifies this Dockerfile and `/health`, if Railway is the selected provider. Provider account settings and billing are outside the code package; inspect them before creating resources.

## Verify the deployed public URL

```sh
curl https://YOUR-SERVICE/health
curl https://YOUR-SERVICE/api/config
```

Then use the UI to:

- Run the default 5% abrupt leak scenario, seed 82001. It should reproduce the packaged example, including CNN–LSTM misses.
- Choose missing pressure sensor; during the dropout every method should suspend scoring, then warm up again.
- Export CSV and verify it includes all 720 observations, not only anomalies.
- Open Results, compare both calibration settings, and download the research report.
- Use a phone-sized viewport and confirm the controls and plots fit.

For scripted browser validation:

```sh
python tools/browser_smoke.py --url https://YOUR-SERVICE
```

Do not declare deployment complete on the basis of a successful build alone. Check the public health endpoint and a real inference request.

## State and resource limits

Use one Uvicorn worker. Models are shared immutable sessions; windows and alert history belong to each run. Runs are ephemeral, capped at 12 and expire after one hour; server restarts discard them. Do not add workers or replicas and expect those run URLs to share state. A durable, authenticated deployment needs a separate state store and upload policy.

Default compute bounds: one simultaneous experiment; 20 new computations per client per ten minutes; 512 kB request bodies; 32–1,200 uploaded readings. Busy requests return 503 and a retry interval. Oversized/invalid requests are rejected rather than silently changing model semantics.

## Publish status

A local commit or ZIP is not a GitHub push, and neither is a live deployment. Publishing requires write access to the repository and access to the chosen hosting service. Verify those steps separately and record the deployed commit/URL when they succeed.

## Checks completed on this update

- Eleven application tests passed.
- Every alarm timestamp in all 264 archived model/run comparisons matched the deployed runtime.
- Desktop and 390 px mobile browser checks passed with zero JavaScript/console errors.
- The exact serving file set and `start.sh` worked with a clean serving-only environment and a non-default `PORT`; a real 720-reading inference request completed.
- The existing Render Python service's `cd backend && uvicorn main:app` entrypoint also passed readiness, dashboard, CSV template and real 720-reading inference checks in a clean serving-only environment.
- Docker was unavailable in the execution environment, so an image build was not performed. The environment could not map another UID; running the image's non-root user is not independently verified here. The Dockerfile specifies that user explicitly.

See `deployment/evidence/validation.json` and the individual reports for the exact scope. Run a separate live-URL smoke test for each deployed commit; local validation alone is not a deployment check.
