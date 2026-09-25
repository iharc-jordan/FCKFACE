"""Local allowlist server for SFace WASM parity; no directory listing or uploads."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--yunet", type=Path, help="Also serve the local YuNet worker parity page")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    own_dir = Path(__file__).resolve().parent
    assets = args.scratch.resolve() / "assets"
    ort = args.scratch.resolve() / "node_modules" / "onnxruntime-web" / "dist"
    allowed = {
        "/": own_dir / "sface_parity.html",
        "/sface_parity.mjs": own_dir / "sface_parity.mjs",
        "/manifest.json": assets / "manifest.json",
        "/case0.f32": assets / "case0.f32",
        "/case1.f32": assets / "case1.f32",
        "/model.onnx": args.model.resolve(),
    }
    if args.yunet:
        allowed.update({
            "/yunet": own_dir / "yunet_parity.html",
            "/yunet_parity.mjs": own_dir / "yunet_parity.mjs",
            "/yunet_worker.mjs": own_dir / "yunet_worker.mjs",
            "/decode_yunet.mjs": own_dir / "decode_yunet.mjs",
            "/yunet-manifest.json": assets / "yunet-manifest.json",
            "/yunet-model.onnx": args.yunet.resolve(),
            "/yunet-cv": own_dir / "yunet_cv_parity.html",
            "/yunet_cv_parity.mjs": own_dir / "yunet_cv_parity.mjs",
            "/yunet_cv_worker.js": own_dir / "yunet_cv_worker.js",
            "/opencv.js": args.scratch.resolve() / "opencv-4.13.0.js",
        })
        for index in (0, 1):
            for suffix in ("source.png", "detector.f32", "aligned.png"):
                name = f"yunet{index}-{suffix}"
                allowed[f"/{name}"] = assets / name
    for name in (
        "ort.min.mjs",
        "ort.min.js",
        "ort-wasm-simd-threaded.mjs",
        "ort-wasm-simd-threaded.wasm",
        "ort-wasm-simd-threaded.jsep.mjs",
        "ort-wasm-simd-threaded.jsep.wasm",
        "ort-wasm-simd-threaded.jspi.mjs",
        "ort-wasm-simd-threaded.jspi.wasm",
        "ort-wasm-simd-threaded.asyncify.mjs",
        "ort-wasm-simd-threaded.asyncify.wasm",
    ):
        allowed[f"/ort/{name}"] = ort / name
    for path in allowed.values():
        if not path.is_file():
            parser.error(f"Missing local parity asset: {path}")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = allowed.get(urlsplit(self.path).path)
            if path is None:
                self.send_error(404)
                return
            content = path.read_bytes()
            suffix = path.suffix
            mime = {
                ".html": "text/html; charset=utf-8",
                ".mjs": "text/javascript; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".json": "application/json",
                ".wasm": "application/wasm",
                ".png": "image/png",
            }.get(suffix, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Local parity page: http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
