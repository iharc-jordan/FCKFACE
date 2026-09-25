"""Allowlist loopback server for two already-used FRLL development crop fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


SFACE_MODEL_SHA = '482fe784b7b4847397459ecf5a0698dfda0bca569815d4a0fc687dd925937fbc'
GHOST_MODEL_SHA = '2132cb7f3bffd75d5de098233e5b5106b15fdf940e824e782d47a06f6e7248f1'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures-dir', required=True, type=Path)
    parser.add_argument('--sface-model-dir', required=True, type=Path)
    parser.add_argument('--ghost-model-dir', required=True, type=Path)
    parser.add_argument('--tfjs-dist', required=True, type=Path)
    parser.add_argument('--port', type=int, default=8768)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    fixtures = args.fixtures_dir.resolve(strict=True)
    sface_dir = args.sface_model_dir.resolve(strict=True)
    ghost_dir = args.ghost_model_dir.resolve(strict=True)
    paths = {
        '/': here / 'combined_crop_parity.html',
        '/combined_crop_parity.mjs': here / 'combined_crop_parity.mjs',
        '/combined_crop_worker.js': here / 'combined_crop_worker.js',
        '/tfjs/tf.min.js': args.tfjs_dist.resolve(strict=True) / 'tf.min.js',
        '/manifest.json': fixtures / 'manifest.json',
        '/sface-reference.json': fixtures / 'sface-reference.json',
        '/ghost-reference.json': fixtures / 'ghost-reference.json',
        '/sface/model.json': sface_dir / 'model.json',
        '/ghost/model.json': ghost_dir / 'model.json',
    }
    for path in paths.values():
        if not path.is_file():
            parser.error(f'Missing asset: {path}')
    manifest = json.loads(paths['/manifest.json'].read_text(encoding='utf-8'))
    sface_ref = json.loads(paths['/sface-reference.json'].read_text(encoding='utf-8'))
    ghost_ref = json.loads(paths['/ghost-reference.json'].read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or \
            [case['identity'] for case in manifest['cases']] != ['frll-024', 'frll-036'] or \
            [case['split'] for case in manifest['cases']] != ['development', 'development']:
        parser.error('Wrong development crop manifest')
    if sface_ref.get('crop_manifest_sha256') != sha(paths['/manifest.json']) or \
            ghost_ref.get('crop_manifest_sha256') != sha(paths['/manifest.json']):
        parser.error('Native references do not match crop manifest')
    if sha(paths['/sface/model.json']) != SFACE_MODEL_SHA or \
            sha(paths['/ghost/model.json']) != GHOST_MODEL_SHA:
        parser.error('Converted model manifest hash changed')
    if json.loads(paths['/sface/model.json'].read_text())['format'] != 'graph-model' or \
            json.loads(paths['/ghost/model.json'].read_text())['format'] != 'layers-model':
        parser.error('Wrong converted model format')
    for case in manifest['cases']:
        for model_name in ('sface', 'ghost'):
            asset = case['crops'][model_name]
            if Path(asset['name']).name != asset['name'] or not asset['name'].endswith('.png'):
                parser.error('Unsafe crop path')
            path = fixtures / asset['name']
            if not path.is_file() or sha(path) != asset['png_sha256']:
                parser.error(f'Crop checksum mismatch: {path}')
            paths[f'/crops/{asset["name"]}'] = path
    for model_name, directory in (('sface', sface_dir), ('ghost', ghost_dir)):
        model = json.loads((directory / 'model.json').read_text())
        for group in model['weightsManifest']:
            for name in group['paths']:
                if Path(name).name != name or not name.endswith('.bin'):
                    parser.error(f'Unsafe model shard path: {name}')
                path = directory / name
                if not path.is_file():
                    parser.error(f'Missing model shard: {path}')
                paths[f'/{model_name}/{name}'] = path
    meta = json.dumps({
        'fixture_manifest_sha256': sha(paths['/manifest.json']),
        'sface_reference_sha256': sha(paths['/sface-reference.json']),
        'ghost_reference_sha256': sha(paths['/ghost-reference.json']),
        'sface_model_json_sha256': SFACE_MODEL_SHA,
        'ghost_model_json_sha256': GHOST_MODEL_SHA,
        'tfjs_sha256': sha(paths['/tfjs/tf.min.js']),
        'model_shards_local_sha256': {name: sha(path) for name, path in paths.items()
                                       if name.startswith(('/sface/', '/ghost/')) and name.endswith('.bin')},
    }).encode('utf-8')

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = urlsplit(self.path).path
            if route == '/probe-meta.json':
                content, mime = meta, 'application/json'
            elif route in paths:
                path = paths[route]
                content = path.read_bytes()
                mime = {'.html': 'text/html; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
                        '.js': 'text/javascript; charset=utf-8', '.json': 'application/json',
                        '.png': 'image/png', '.bin': 'application/octet-stream'}.get(path.suffix, 'application/octet-stream')
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
    print(f'Local development-crop component check: http://127.0.0.1:{args.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
