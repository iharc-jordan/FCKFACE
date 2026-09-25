# Portrait-renderer reproducibility

These are development appearance experiments, not a validated privacy method. The
four prespecified FRLL development identities are `frll-001`, `frll-003`,
`frll-083`, and `frll-143`. H3c's four-level surface and H3d's cyan-paper
handmade contours were rejected at root appearance screening before recognition
scoring because individual likeness was uncertain. Root screening is not an
independent human study. The author-model render is an established prior-art
comparison, not a FCKFACE invention. Its eight fixed JPEGs produced zero valid
all-seven-condition SFace gallery nonmatches; two `frll-143` outputs were
inconclusive because YuNet found no face.

The learned render uses Caroline Chan, Frédo Durand, and Phillip Isola's
[Informative Drawings](https://github.com/carolineec/informative-drawings)
`Generator(3, 1, 3)` and the two weights in the
[author's demo Space](https://huggingface.co/spaces/carolineec/informativedrawings/tree/main).
Their [paper](https://arxiv.org/abs/2203.12691) concerns drawing geometry and
semantics, not preservation of facial identity or resistance to recognition.
The GitHub source has an MIT `LICENSE`; the Space declares `license: mit`, but
the checkpoints have no separate explicit license file. Review artifact rights
before redistributing weights. Keep the source, checkpoints, personal images,
embeddings, and all run outputs outside this repository.

Pin and verify these official files. The first three paths below are relative
to an external `informative-drawings` directory.

| File | Official revision | SHA-256 |
| --- | --- | --- |
| `source/model.py` | GitHub `2349aee4daf7cb01d8de645b0bbb4f4392fd1395` | `e2ce57dea009804343e3a4744a604052b113721bd4770378d75e63b8263cf313` |
| `source/LICENSE` | Same GitHub commit | `865d995c72673df6f3e6e1d135b7c412ca9d8793d34d5577e6089f8994971f6d` |
| `author-space-source/app.py` | Space `bd4b4299be505803e036203a39c02024b4cfee11` | `4e989aff7a9f4491e9bf463dff0b7c35f84b52b34a0bf4ca918b525fa7d593d0` |
| `author-space-weights/model.pth` | Same Space commit | `c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6` |
| `author-space-weights/model2.pth` | Same Space commit | `30a534781061f34e83bb9406b4335da4ff2616c95d22a585c1245aa8363e74e0` |

For example, from the repository root in PowerShell, choose a private absolute
`$data` directory and acquire the small pinned author files:

```powershell
$data = 'D:\private\FCKFACE-data'
$author = Join-Path $data 'informative-drawings'
New-Item -ItemType Directory -Force -Path $author | Out-Null
git clone https://github.com/carolineec/informative-drawings.git (Join-Path $author 'source')
git -C (Join-Path $author 'source') checkout --detach 2349aee4daf7cb01d8de645b0bbb4f4392fd1395
New-Item -ItemType Directory -Force -Path (Join-Path $author 'author-space-source'),(Join-Path $author 'author-space-weights') | Out-Null
$space = 'https://huggingface.co/spaces/carolineec/informativedrawings'
$revision = 'bd4b4299be505803e036203a39c02024b4cfee11'
Invoke-WebRequest "$space/raw/$revision/app.py" -OutFile (Join-Path $author 'author-space-source/app.py')
Invoke-WebRequest "$space/resolve/$revision/model.pth?download=true" -OutFile (Join-Path $author 'author-space-weights/model.pth')
Invoke-WebRequest "$space/resolve/$revision/model2.pth?download=true" -OutFile (Join-Path $author 'author-space-weights/model2.pth')
Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $author 'source/model.py'),(Join-Path $author 'source/LICENSE'),(Join-Path $author 'author-space-source/app.py'),(Join-Path $author 'author-space-weights/model.pth'),(Join-Path $author 'author-space-weights/model2.pth')
```

The two checkpoints are 17,173,511 bytes each. `render_learned_portrait.py`
checks their digests and the source digests directly; no private
`provenance.json` is required. Use a Python environment with NumPy, OpenCV,
Pillow, CPU-capable PyTorch and torchvision. The measured run used Torch
2.14.0+cpu, torchvision 0.29.0 and Pillow 12.3.0. Do not execute the Space's
`app.py`: it launches Gradio on import. The wrapper imports the pinned GitHub
`model.py`, loads each state dict with `weights_only=True`, and follows the
author's CPU RGB-to-tensor / grayscale-output path. The GitHub `test.py`
unconditionally calls CUDA and is not the entry point used here.

Prepare the external FRLL `manifest.json` as described in
[research/README.md](README.md), then calibrate SFace with the 20 separate
calibration identities. Acquire YuNet and SFace from the official links and
verify hashes in [models.json](models.json). The following placeholders are
absolute private paths; each output directory must be new:

```powershell
$manifest = Join-Path $data 'frll/manifest.json'
$yunet = Join-Path $data 'models/face_detection_yunet_2023mar.onnx'
$sface = Join-Path $data 'models/face_recognition_sface_2021dec.onnx'
python research/calibrate_sface.py --manifest $manifest --output (Join-Path $data 'runs/calibration-sface-v1') --yunet $yunet --sface $sface
$calibration = Join-Path $data 'runs/calibration-sface-v1/calibration.json'
$stencil = Join-Path $data 'runs/portrait-stencil-v1'
python research/render_stencil.py --manifest $manifest --calibration $calibration --yunet $yunet --sface $sface --out $stencil
python research/render_contours.py --manifest $manifest --calibration $calibration --yunet $yunet --sface $sface --out (Join-Path $data 'runs/contour-portrait-v1')
python research/render_learned_portrait.py --manifest $manifest --source-preview (Join-Path $stencil 'previews.json') --author $author --out (Join-Path $data 'runs/informative-portrait-v1')
```

The stencil preview supplies source-only face boxes. The learned wrapper checks
its 32 rows, the exact four identities, eight fixed arm/preset combinations per
identity, source image hashes, and consistent in-frame boxes. It ignores old
machine-specific `source_path` text and resolves images from the supplied FRLL
manifest. For an exact rerun of one existing private preview, add
`--expected-source-preview-sha256` with its lowercase 64-character digest.

The already executed learned preview remains immutable under its external
`runs/informative-portrait-v1` directory. Its saved source snapshot has SHA-256
`fa211d82bc8ab80f430867283f74753cba8497d060264544c9699e30f37218d1`.
The current wrapper changes only acquisition and preflight checks; its 512-pixel
crop, model loading, grayscale face composite, JPEG export, and experiment
settings remain the same. No inference was rerun for this portability edit.
