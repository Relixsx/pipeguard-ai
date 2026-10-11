"""Exercise the actual dashboard and API in Chromium, with no mocked scores.

Use --start-server for local QA, or --url for a deployed service.
PIPEGUARD_CHROME optionally selects an existing compatible Chrome executable.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--server-python", default=sys.executable)
    parser.add_argument("--output", default=str(Path(tempfile.gettempdir()) / "pipeguard-browser-qa"))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    process, log = None, None
    if args.start_server:
        # A clean serving environment must not inherit the training PYTHONPATH.
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        log = (output / "server.log").open("w")
        process = subprocess.Popen([args.server_python,"-m","uvicorn","backend.main:app",
                                    "--host","127.0.0.1","--port","8000"],
                                   cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        for attempt in range(60):
            try:
                with urllib.request.urlopen(args.url + "/health", timeout=2) as response:
                    if json.load(response)["status"] == "ready":
                        break
            except Exception:
                if process and process.poll() is not None:
                    raise RuntimeError("Server startup failed; inspect server.log")
                time.sleep(.25)
        else:
            raise RuntimeError("Server did not become ready")
        errors, checks = [], []
        with sync_playwright() as playwright:
            launch = {"headless":True,"args":["--no-sandbox","--disable-dev-shm-usage"]}
            if os.getenv("PIPEGUARD_CHROME"):
                launch["executable_path"] = os.environ["PIPEGUARD_CHROME"]
            browser = playwright.chromium.launch(**launch)
            page = browser.new_page(viewport={"width":1440,"height":1000}, device_scale_factor=1)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
            page.goto(args.url, wait_until="networkidle")
            expect(page.locator("#server-status")).to_have_text("5 methods ready")
            expect(page.locator("#run-kind")).to_have_text("SIMULATED READINGS")
            expect(page.locator("#pressure-chart svg")).to_be_visible()
            assert page.locator("#score-chart svg path").count() >= 7
            assert page.locator("#summary-cnn_lstm").inner_text() == "No new post-onset alarm so far"
            checks.append("Packaged example displays actual five-model inference")
            page.screenshot(path=output / "desktop-lab.png", full_page=True)
            scores = page.locator(".score-value").all_inner_texts()
            page.locator("#show-truth").uncheck()
            assert page.locator(".score-value").all_inner_texts() == scores
            assert page.locator(".volume.injected").count() == 0
            page.locator("#show-truth").check()
            page.locator("#play-button").click()
            page.wait_for_timeout(400)
            assert page.locator("#clock").inner_text() != "59:55"
            page.locator("#play-button").click()
            checks.append("Truth toggle cannot alter model scores; replay controls work")
            page.locator("#tab-results").click()
            expect(page.locator("#result-table tr")).to_have_count(4)
            expect(page.locator("#result-table tr").first).to_contain_text("36 / 36")
            page.locator("#coverage-narrow").click()
            assert "36 / 36" != page.locator("#result-table tr").first.locator("td").nth(1).inner_text()
            page.locator("#coverage-broad").click()
            page.screenshot(path=output / "desktop-results.png", full_page=True)
            checks.append("Both calibration comparisons and saved acoustic results display")
            page.locator("#tab-research").click()
            expect(page.locator("#references li")).to_have_count(10)
            checks.append("Ten paper references and qualified dataset plan display")
            page.locator("#tab-methods").click()
            expect(page.locator("#network-output .method-chart svg")).to_be_visible()
            expect(page.locator("#recorded-transfer")).to_contain_text("684")
            page.locator("#network-scenario").select_option("weak_excitation")
            with page.expect_response(lambda r: r.url.endswith('/api/labs/network')) as network:
                page.locator("#network-form button").click()
            assert network.value.status == 201
            expect(page.locator("#network-output .guard-status")).to_contain_text("Attribution withheld")
            with page.expect_download() as lab_download:
                page.locator("#network-output a").click()
            lab_download.value.save_as(output / 'network.json')
            assert not any(r['supported'] for r in json.loads((output / 'network.json').read_text())['rolling'])
            page.locator("#information-excitation").select_option("wrong_location")
            with page.expect_response(lambda r: r.url.endswith('/api/labs/information')) as information:
                page.locator("#information-form button").click()
            assert information.value.status == 201
            expect(page.locator("#information-output .guard-status")).to_contain_text("fails")
            page.locator("#acoustic-scenario").select_option("correlated")
            with page.expect_response(lambda r: r.url.endswith('/api/labs/acoustic')) as acoustic:
                page.locator("#acoustic-form button").click()
            assert acoustic.value.status == 201
            expect(page.locator("#acoustic-output .guard-status")).to_contain_text("unsupported")
            from backend.paper_labs import acoustic_data
            acoustic_rows = acoustic_data(96001, samples=1536)['measurements']
            acoustic_csv = 'm1,m2,m3,m4,m5\n'+'\n'.join(','.join(map(str, row)) for row in acoustic_rows)
            page.locator('.method-upload summary').click()
            with page.expect_response(lambda r: r.url.endswith('/api/labs/acoustic/analyze')) as uploaded_acoustic:
                page.locator('#acoustic-upload').set_input_files({'name': 'channels.csv', 'mimeType': 'text/csv', 'buffer': acoustic_csv.encode()})
            assert uploaded_acoustic.value.status == 201
            assert uploaded_acoustic.value.json()['lab'] == 'acoustic_upload'
            assert uploaded_acoustic.value.json()['rms_sample_indices'][0] >= 1024
            expect(page.locator('#acoustic-output .rms-heatmap svg')).to_have_count(2)
            checks.append('Synchronized acoustic CSV upload fits, verifies and monitors disjoint segments through the actual API')
            page.screenshot(path=output / "desktop-methods.png", full_page=True)
            checks.append("Actual paper APIs, weak-excitation guard, excitation-location failure, correlated-source screen and JSON export work")
            page.locator("#tab-lab").click()
            page.locator("#scenario").select_option("missing_sensor")
            with page.expect_response(lambda response: response.url.endswith("/api/runs") and response.request.method == "POST") as result:
                page.locator("#run-button").click()
            assert result.value.status == 201
            expect(page.locator("#message")).to_contain_text("Experiment complete")
            page.locator("#timeline").evaluate("el => { el.value='290'; el.dispatchEvent(new Event('input',{bubbles:true})); }")
            assert all("Sensor data missing" in text for text in page.locator(".model-status").all_inner_texts())
            page.locator("#timeline").evaluate("el => { el.value='340'; el.dispatchEvent(new Event('input',{bubbles:true})); }")
            assert all("Warming up" in text for text in page.locator(".model-status").all_inner_texts())
            checks.append("New inference request, dropout suspension and warm-up display")
            with page.expect_download() as download:
                page.locator("#export-csv").click()
            export = output / "export.csv"
            download.value.save_as(export)
            assert len(export.read_text().strip().splitlines()) == 721
            template = page.request.get(args.url + "/api/template.csv").body()
            with page.expect_response(lambda response: response.url.endswith("/api/analyze")) as analysis:
                page.locator("#csv-upload").set_input_files({"name":"sensor.csv","mimeType":"text/csv","buffer":template})
            assert analysis.value.status == 201
            expect(page.locator("#run-kind")).to_have_text("UPLOADED READINGS")
            expect(page.locator("#truth-callout")).to_contain_text("No ground truth supplied")
            checks.append("CSV download contains all readings; SI-schema import uses actual API")
            page.locator("#csv-upload").set_input_files({"name":"bad.csv","mimeType":"text/csv","buffer":b"timestamp_s,pressure\n0,3.9"})
            expect(page.locator("#message")).to_contain_text("CSV requires these columns")
            # Small screens use the same real app, with no separate mock.
            page.set_viewport_size({"width":390,"height":844})
            page.reload(wait_until="networkidle")
            expect(page.locator("#server-status")).to_have_text("5 methods ready")
            page.evaluate("document.activeElement.blur(); window.scrollTo(0,0)")
            page.screenshot(path=output / "mobile-lab.png", full_page=True)
            overflow = page.evaluate("Array.from(document.querySelectorAll('body *')).filter(el=>el.getBoundingClientRect().right>window.innerWidth+1).map(el=>({tag:el.tagName,id:el.id,class:el.className,right:el.getBoundingClientRect().right})).slice(0,15)")
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), f"Mobile horizontal overflow: {overflow}"
            page.locator("#tab-methods").click()
            page.screenshot(path=output / "mobile-methods.png", full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Paper-method panel overflows mobile screen"
            checks.append("390 px responsive layout has no horizontal overflow")
            assert not errors, errors
            browser.close()
        report = {"checks":checks,"browser_errors":errors,"status":"passed","url":args.url}
        (output / "browser-report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2))
    finally:
        if process:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill()
        if log: log.close()


if __name__ == "__main__":
    main()
