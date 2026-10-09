"""Verify the Docker file set and entrypoint without requiring a Docker daemon.

This is a non-root application startup check, not a container image build.
Provide a Python environment containing requirements.txt only.
"""
import argparse
import errno
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    python = Path(args.python).absolute()
    with tempfile.TemporaryDirectory(prefix="pipeguard-startup-") as folder:
        stage = Path(folder)
        stage.chmod(0o755)
        for name in ("backend", "frontend", "deployment"):
            shutil.copytree(ROOT / name, stage / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(ROOT / "start.sh", stage / "start.sh")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.update({"PATH":str(python.parent)+os.pathsep+os.defpath,"PORT":"41077",
                    "PYTHONDONTWRITEBYTECODE":"1","PYTHONUNBUFFERED":"1"})
        # Refuse accidentally testing an environment containing training packages.
        clean = subprocess.run([str(python),"-c","import importlib.util; assert importlib.util.find_spec('torch') is None"],
                               env=env,capture_output=True,text=True)
        if clean.returncode:
            raise RuntimeError("Provide a serving-only Python environment without Torch")
        options = {"user":65534,"group":65534} if os.getuid() == 0 else {}
        with (stage / "startup.log").open("w") as log:
            try:
                process = subprocess.Popen(["sh","start.sh"],cwd=stage,env=env,stdout=log,stderr=subprocess.STDOUT,**options)
            except OSError as error:
                if not options or error.errno not in (errno.EINVAL, errno.EPERM):
                    raise
                # Some managed runtimes map only UID 0. Report that limitation.
                options = {}
                process = subprocess.Popen(["sh","start.sh"],cwd=stage,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                for attempt in range(80):
                    try:
                        with urllib.request.urlopen("http://127.0.0.1:41077/health",timeout=1) as response:
                            health = json.load(response)
                        break
                    except Exception:
                        if process.poll() is not None:
                            raise RuntimeError((stage / "startup.log").read_text())
                        time.sleep(.1)
                else:
                    raise RuntimeError("Entrypoint did not become ready")
                assert health["models_loaded"] == 4
                for path in ("/", "/assets/app.js", "/api/example", "/api/evidence", "/api/download/report"):
                    assert urllib.request.urlopen("http://127.0.0.1:41077"+path,timeout=5).status == 200
                request = urllib.request.Request("http://127.0.0.1:41077/api/runs",data=json.dumps({"scenario":"sensor_bias","seed":84010}).encode(),headers={"Content-Type":"application/json"})
                with urllib.request.urlopen(request,timeout=20) as response:
                    run = json.load(response)
                assert len(run["readings"]) == 720
                assert run["ground_truth"]["onset_s"] is None
                result = {"status":"passed","serving_environment_has_torch":False,
                          "non_root_verified":bool(options) or os.getuid()!=0,"dynamic_port":41077,
                          "methods_loaded":4,"actual_inference_readings":720,
                          "scope":"Docker file-set/entrypoint check; no Docker image build performed"}
                if args.output: Path(args.output).write_text(json.dumps(result,indent=2)+"\n")
                print(json.dumps(result,indent=2))
            finally:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill()


if __name__ == "__main__":
    main()
