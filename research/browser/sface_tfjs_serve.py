"""Allowlist loopback server for the synthetic SFace TF.js GraphModel probe."""

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
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--synthetic-rgb', required=True, type=Path)
    parser.add_argument('--tfjs-dist', required=True, type=Path)
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    model_dir = args.model_dir.resolve()
    model = model_dir / 'model.json'
    reference = args.reference.resolve()
    rgb = args.synthetic_rgb.resolve()
    tfjs = args.tfjs_dist.resolve() / 'tf.min.js'
    allowed = {
        '/': here / 'sface_tfjs_parity.html',
        '/sface_tfjs_parity.mjs': here / 'sface_tfjs_parity.mjs',
        '/sface_tfjs_worker.js': here / 'sface_tfjs_worker.js',
        '/tfjs/tf.min.js': tfjs,
        '/cosine-reference.json': reference,
        '/synthetic-rgb.bin': rgb,
        '/model/model.json': model,
    }
    for path in allowed.values():
        if not path.is_file():
            parser.error(f'Missing asset: {path}')
    manifest = json.loads(model.read_text(encoding='utf-8'))
    if manifest.get('format') != 'graph-model':
        parser.error('Expected a standard TF.js GraphModel')
    if manifest['signature']['inputs']['data']['name'] != 'data:0' or \
            manifest['signature']['outputs']['output_0']['name'] != 'Identity:0':
        parser.error('Unexpected GraphModel signature')
    for group in manifest['weightsManifest']:
        for name in group['paths']:
            if Path(name).name != name or not name.endswith('.bin'):
                parser.error(f'Unsafe shard path: {name}')
            allowed[f'/model/{name}'] = model_dir / name
    for path in allowed.values():
        if not path.is_file():
            parser.error(f'Missing asset: {path}')
    frozen = json.loads(reference.read_text(encoding='utf-8'))
    if frozen.get('schema_version') != 1 or frozen.get('model_sha256') != \
            '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79' or \
            digest(rgb) != frozen.get('rgb_u8_sha256'):
        parser.error('Wrong frozen synthetic reference or RGB tensor')
    meta = json.dumps({
        'model_sha256': digest(model), 'reference_sha256': digest(reference),
        'tfjs_sha256': digest(tfjs),
        'weight_shards_sha256': {name.removeprefix('/model/'): digest(path)
                                 for name, path in allowed.items()
                                 if name.startswith('/model/') and name.endswith('.bin')},
    }).encode('utf-8')

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = urlsplit(self.path).path
            if route == '/probe-meta.json':
                content, mime = meta, 'application/json'
            elif route in allowed:
                path = allowed[route]
                content = path.read_bytes()
                mime = {'.html': 'text/html; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
                        '.js': 'text/javascript; charset=utf-8', '.json': 'application/json',
                        '.bin': 'application/octet-stream'}.get(path.suffix, 'application/octet-stream')
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(content)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Local synthetic SFace TF.js probe: http://127.0.0.1:{args.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
