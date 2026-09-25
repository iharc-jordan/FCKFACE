"""Serve only a local GhostFaceNet TF.js synthetic parity probe and its assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tfjs-dist", type=Path, required=True)
    parser.add_argument("--variant", choices=("author", "float32"), required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    model = args.model_dir.resolve() / "model.json"
    reference = args.reference.resolve()
    tfjs = args.tfjs_dist.resolve() / "tf.min.js"
    allowed = {
        "/": here / "ghost_tfjs_parity.html",
        "/ghost_tfjs_parity.mjs": here / "ghost_tfjs_parity.mjs",
        "/ghost_tfjs_worker.js": here / "ghost_tfjs_worker.js",
        "/ghost_tfjs_warm_worker.js": here / "ghost_tfjs_warm_worker.js",
        "/tfjs/tf.min.js": tfjs,
        "/float32-reference.json": reference,
        "/model/model.json": model,
    }
    if not model.is_file():
        parser.error(f"Model JSON missing: {model}")
    model_json = json.loads(model.read_text(encoding="utf-8"))
    if model_json.get("format") != "layers-model":
        parser.error("Expected standard TF.js Layers model")
    for group in model_json["weightsManifest"]:
        for name in group["paths"]:
            if Path(name).name != name or not name.endswith(".bin"):
                parser.error(f"Unsafe shard path: {name}")
            allowed[f"/model/{name}"] = args.model_dir.resolve() / name
    for path in allowed.values():
        if not path.is_file():
            parser.error(f"Missing asset: {path}")
    frozen = json.loads(reference.read_text(encoding="utf-8"))
    if frozen.get("schema_version") != 1 or frozen.get("input_shape") != [1, 112, 112, 3]:
        parser.error("Wrong synthetic reference schema")
    meta = json.dumps({
        "variant": args.variant,
        "model_sha256": digest(model),
        "reference_sha256": digest(reference),
        "tfjs_sha256": digest(tfjs),
        "weight_shards_sha256": {name.removeprefix("/model/"): digest(path)
                                 for name, path in allowed.items() if name.startswith("/model/") and name.endswith(".bin")},
    }).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = urlsplit(self.path).path
            if route == "/probe-meta.json":
                content, mime = meta, "application/json"
            elif route in allowed:
                path = allowed[route]
                content = path.read_bytes()
                mime = {
                    ".html": "text/html; charset=utf-8",
                    ".mjs": "text/javascript; charset=utf-8",
                    ".js": "text/javascript; charset=utf-8",
                    ".json": "application/json",
                    ".bin": "application/octet-stream",
                }.get(path.suffix, "application/octet-stream")
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Local {args.variant} synthetic probe: http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
