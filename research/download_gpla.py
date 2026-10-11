"""Download and verify the actual GPLA-12 v1 release, not the v3 placeholder.

Third-party recordings are not committed. Repository license is retained;
recording-session independence and microphone synchronization are unresolved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT='afec271f4eb445a976826c6a45485892e5fbab53'
REPOSITORY='Deep-AI-Application-DAIP/acoustic-leakage-dataset-GPLA-12'
FILES={'data/data_v1/data.csv':'00135bf5f4a417d07219dd9cd340dd3834dd2cbb75aea7b11fdacfba2f0fcba1',
       'data/data_v1/label.csv':'97235dd9abdb0abf3082bb310c96ca300f276b948e15dacf9465b4ef20997332',
       'LICENSE':'8c349f80764d0648e645f41ef23772a70c995a0924b5235f735f4a3d09df127c'}


def download(directory):
    directory.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,expected in FILES.items():
        url=f'https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{name}'
        with urllib.request.urlopen(url,timeout=60) as response:blob=response.read()
        digest=hashlib.sha256(blob).hexdigest()
        if digest!=expected:raise RuntimeError(f'Source checksum mismatch: {name}')
        dest=directory/Path(name).name;dest.write_bytes(blob)
        records.append({'source_path':name,'sha256':digest,'bytes':len(blob),'url':url})
    result={'repository':REPOSITORY,'commit':COMMIT,'files':records,
            'version':'v1 CSV; v3 folder contains a one-byte placeholder',
            'scope':'684 individual gas-acoustic recordings; no healthy class or synchronized-channel contract'}
    (directory/'acquisition.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,default=Path('data/gpla'))
    args=parser.parse_args();print(json.dumps(download(args.data_dir),indent=2))
