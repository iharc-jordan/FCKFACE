# ArcFace ONNX native input-gradient component check

The already cached InsightFace `buffalo_l` development recognizer `w600k_r50.onnx` was tested as a **network component**, with no new detection, gallery comparison, edited image, optimizer, or browser run. The official model SHA-256 was `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` (174,383,860 bytes). A verified ordinary private copy under `Downloads/FCKFACE-data/runs/arcface-onnx-gradient-v1/` was the sole input to direct `onnx2torch.convert(onnx.load(...))`; both original and copy retained that hash afterward. The [development adapter](arcface_development.py) and [pinned InsightFace source](https://github.com/deepinsight/insightface/tree/1480e705287bc5d59f923b46c260ec6e3e4150f6) establish the official preprocessing: BGR crop, `swapRB=True`, and `(RGB−127.5)/127.5`. The Torch check applied that normalization to raw RGB input so its gradient is measured per raw RGB pixel. The pretrained weights remain private under [InsightFace's noncommercial research terms](https://github.com/deepinsight/insightface/blob/1480e705287bc5d59f923b46c260ec6e3e4150f6/README.md); MIT code terms do not authorize public redistribution of the weights.

The three synthetic 112×112 RGB inputs were a modulo-256 ramp, flat 128, and seed-0 random pixels. The other two were the already reviewed `frll-024` and `frll-036` **SFace-aligned** 112-pixel PNG crops from `combined-crop-browser-v1/manifest.json` (SHA-256 `c2f8885be4b2808a2c3ec6606a42cbadee90f10ee531cec781b3e2dbb4789162`). Their PNG hashes were `d30f36ab883fbe1ae63b7cf0506c8c7c8454d9c52483cb1cafa03a8d81c291cf` and `d4037b2aff1d28ec6622c880532e107d83dc6a7f4e96a18f974a47bb8854f31f`. They are valid network input fixtures, **not** official ArcFace/SCRFD-aligned crops. The private frozen `protocol-v2.json` (SHA-256 `c2776514e095fa0e891f9104b9843f681c4aaa3465a97aece3c52cd0df449088`) pins their decoded-pixel and synthetic tensor hashes, model, source, seed, and limits before inference. The earlier unrun `protocol.json`/source snapshot remain preserved separately at SHA-256 `2a0a389895cd9d61d08e4f291d5e0079eea08ce423b320cf12b670b25322e880` / `d6c820dd08b402dc6727be0ce0e4e79682adddaeaf52c0fdf998d75bb3c42c26`.

The frozen check required raw ONNX Runtime versus Torch maximum absolute error ≤`1e-3`, normalized cosine ≥`.99999`, a finite nonzero gradient of a seed-7095 random-unit-target cosine loss, and central ONNX Runtime finite differences at the three largest nontrivial **interior** gradient pixels per input. Epsilon was `.1` raw pixel (`.1/127.5` normalized); at least two per input had to pass relative error ≤`.1`, with every coordinate passing relative ≤`.1` or absolute ≤`1e-6`. All five forward checks and all **15/15 nontrivial relative finite differences** passed on the first run:

| Input | Raw max abs | Unit cosine | Gradient L2 | Largest FD relative error |
| --- | ---: | ---: | ---: | ---: |
| Range | `3.979e-6` | `.999999999995715` | `.000478774` | `.016573` |
| Flat | `5.752e-6` | `.999999999989326` | `.001198580` | `.013323` |
| Random | `3.457e-6` | `.999999999995278` | `.000655517` | `.010350` |
| `frll-024` crop | `3.874e-6` | `.999999999998594` | `.001763622` | `.007282` |
| `frll-036` crop | `6.676e-6` | `.999999999998530` | `.001628248` | `.014783` |

Direct conversion took **1.071 seconds**; the script measured **7.455 seconds** for its checked work and peak process working set **1,197,150,208 bytes (1.20 GB)**. The shell-observed wall time was **18.68 seconds**, including Python/package startup. These are native CPU measurements, not browser latency. The result's postflight hashes confirm the source, protocol, original/private model, crop manifest, and both PNGs stayed unchanged. Private `component-v2/results.json` is SHA-256 `ac5a8a994551fd1befbe573f601e1205ab4699a3d3d72a00b4bd5f460a243eb7`; `component-arrays.npz` is `b97be430081b26dda12c6af83af642b18f47c1e70b15feec3c12369369c567a2`; `preflight.json` is `3abfc6c371701c69a2d4deaed7259598b8cfc32de7d4a9ec89b2cd20c39448fa`. [The executed source](arcface_onnx_gradient.py) and its frozen v2 copy are SHA-256 `ff2adad7e5f3e2eeb46a34ea355fb6c344ff089424236a688eacd4c9465ae2df`.

To reproduce from the preserved frozen inputs into a **new output directory** after verifying their hashes, use the existing read-only Python 3.12.14 environment with `onnx2torch==1.5.15`, `torch==2.14.0+cpu`, `onnx==1.23.0`, `onnxruntime==1.30.0`, NumPy `2.5.3`, and Pillow. Keep the private ordinary model copy and protocol outside Git, then invoke:

```powershell
$data = 'PATH_TO_PRIVATE_FCKFACE_DATA'
$py = 'PATH_TO_EXISTING_READONLY_PYTHON_EXE'
$run = Join-Path $data 'runs/arcface-onnx-gradient-v1'
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
& $py research/arcface_onnx_gradient.py `
  --protocol (Join-Path $run 'protocol-v2.json') `
  --original-model (Join-Path $data 'arcface-development-v1/w600k_r50.onnx') `
  --private-model (Join-Path $run 'private-w600k_r50.onnx') `
  --crop-manifest (Join-Path $data 'runs/combined-crop-browser-v1/manifest.json') `
  --crops (Join-Path $data 'runs/combined-crop-browser-v1') `
  --output (Join-Path $run 'component-new')
```

The frozen v2 source refuses changed hashes or an existing output directory. This verifies only the differentiable recognizer on fixed crops; it does not prove an ArcFace photo-to-crop gradient, any edited-image transfer, a product runtime, or a privacy result.
