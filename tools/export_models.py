"""Export the actual research checkpoints; verify batch sizes and score parity.

Run from the repository root with the research dependencies installed.
The web server only needs ONNX Runtime, not PyTorch.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from pipeguard.models import CNNLSTMAutoencoder  # noqa: E402


def main():
    torch.set_num_threads(1)
    output = ROOT / "deployment/models"
    checks = []
    for name in ("cnn_lstm", "hybrid"):
        source = ROOT / f"research/results/models/{name}_11.pt"
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        model = CNNLSTMAutoencoder(checkpoint["input_dim"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        target = output / f"{name}_11.onnx"
        dummy = torch.zeros(1, 30, checkpoint["input_dim"])
        torch.onnx.export(
            model, dummy, str(target), input_names=["window"],
            output_names=["reconstruction"], opset_version=17, dynamo=False,
            dynamic_axes={"window": {0: "batch"}, "reconstruction": {0: "batch"}},
        )
        onnx.checker.check_model(onnx.load(target))
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(target), options, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(42)
        for batch in (1, 7, 32):
            x = rng.normal(size=(batch, 30, checkpoint["input_dim"])).astype(np.float32)
            with torch.no_grad():
                expected = model(torch.from_numpy(x)).numpy()
            actual = session.run(None, {"window": x})[0]
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
            score_expected = np.mean((expected - x) ** 2, axis=(1, 2))
            score_actual = np.mean((actual - x) ** 2, axis=(1, 2))
            np.testing.assert_allclose(score_actual, score_expected, rtol=2e-6, atol=2e-6)
            checks.append({"model": name, "batch": batch,
                           "max_output_error": float(np.max(np.abs(actual - expected))),
                           "max_score_error": float(np.max(np.abs(score_actual - score_expected)))})
    manifest = {
        "format": "ONNX", "opset": 17, "seed": 11,
        "torch_version": torch.__version__, "onnxruntime_version": ort.__version__,
        "checks": checks,
        "files": {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                            "bytes": p.stat().st_size}
                  for p in output.iterdir() if p.suffix in (".onnx", ".npz", ".json")
                  and p.name != "manifest.json"},
        "checkpoints": {name: hashlib.sha256((ROOT / f"research/results/models/{name}_11.pt").read_bytes()).hexdigest()
                        for name in ("cnn_lstm", "hybrid")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"verified": checks, "files": manifest["files"]}, indent=2))


if __name__ == "__main__":
    main()
