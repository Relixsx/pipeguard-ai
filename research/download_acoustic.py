"""Download only explicitly listed public gas-pipeline recordings.

No third-party recordings are redistributed with this project. Acquisition
groups and dataset-wide reuse terms are not documented in the source repo.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, hashlib, json, re, time, urllib.request, wave

FOLDERS = {
    "blower": "1YcOF3v87xwiv39uaqPYSTVzlunQBZZVM",
    "hole_blower": "1zMkh8syBjMoBLcEtVx7rpj3oPLyFjPne",
    "valve_blower": "1kiDtmcZHYs-7WjCqeNWdHV8BmhvE3JoS",
}
SOURCE_REPO = "https://github.com/mengdinet/Gas-pipeline-leakage-data-set"

def folder_items(raw):
    match = re.search(r"window\['_DRIVE_ivd'\]\s*=\s*'((?:\\.|[^'])*)'", raw)
    if not match:
        raise ValueError("Public folder listing format changed; no inferred IDs used.")
    encoded = re.sub(r'\\x([0-9a-fA-F]{2})', lambda m: chr(int(m[1], 16)), match[1])
    return json.loads(encoded.replace('\\/', '/').replace("\\'", "'"))[0]

def get_bytes(url):
    request = urllib.request.Request(url, headers={"User-Agent": "PipeGuard-research/1.0"})
    with urllib.request.urlopen(request, timeout=35) as response:
        return response.read()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/acoustic")
    parser.add_argument("--manifest-dir", default=None, help="Optional cached folder-list JSONs")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = Path(args.data_dir)
    items = []
    for category, folder_id in FOLDERS.items():
        cached = Path(args.manifest_dir or '.') / f"meng_{category}_manifest.json"
        if args.manifest_dir and cached.exists():
            rows = json.loads(cached.read_text())
        else:
            rows = folder_items(get_bytes(f"https://drive.google.com/drive/folders/{folder_id}").decode())
        for row in rows:
            if row[3] != "application/vnd.google-apps.folder" and row[2].endswith('.wav'):
                items.append((category, row[2], row[0]))
    def fetch(item):
        category, name, file_id = item
        path = root/category/name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
        for attempt in range(3):
            try:
                payload = path.read_bytes() if path.exists() else get_bytes(url)
                if payload[:4] != b'RIFF' or payload[8:12] != b'WAVE':
                    raise ValueError("Response is not a WAV file")
                path.write_bytes(payload)
                with wave.open(str(path)) as audio:
                    info = dict(channels=audio.getnchannels(), sample_rate=audio.getframerate(),
                                samples=audio.getnframes(), sample_width=audio.getsampwidth())
                return dict(category=category, filename=name, file_id=file_id, url=url,
                            sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload), **info)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1+attempt)
    manifest = []
    with ThreadPoolExecutor(args.workers) as executor:
        jobs = [executor.submit(fetch, item) for item in items]
        for job in as_completed(jobs):
            manifest.append(job.result())
            if len(manifest) % 25 == 0:
                print(f"Verified {len(manifest)}/{len(items)} recordings", flush=True)
    manifest.sort(key=lambda row: (row['category'], row['filename']))
    root.mkdir(parents=True, exist_ok=True)
    (root/'acquisition_manifest.json').write_text(json.dumps({
        'source_repo': SOURCE_REPO, 'recordings': manifest,
        'scope': 'Only items visible in the fetched folder listings; not a claim of full dataset coverage.',
        'license': 'No explicit dataset license found in the linked source repository.',
        'group_warning': 'Filename prefixes are provisional grouping keys, not verified acquisition sessions.'
    }, indent=2))
    print(f"Saved acquisition manifest for {len(manifest)} files", flush=True)

if __name__ == '__main__':
    main()
