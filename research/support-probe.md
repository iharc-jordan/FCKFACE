# H11 in-grid support probe: bounded development result

H11 compared H5's sparse 14×14 coloured dots with their exact **inside-grid complement** to test whether dot support was limiting transfer. The wider inverse support did **not** meet the frozen ArcFace or native-model criteria on the two development people. Retire this three-forward-step setting without expansion or reseeding. It does not rule out other artwork coverage, objectives or optimization budgets, and does not establish a novel protection method.

Both arms used the same 588 RGB cell coefficients, seed-0 `N(0, 0.1)` initialization, Adam learning rate 0.15, face RMS target 16, channel cap 64, feathered face mask and H7 fixed clean landmarks for each condition. The dot arm used H5's literal radius-0.3 circular mask in every 14×14 cell. The inverse arm used `inside_grid AND NOT dot_mask`; it changed **no pixels outside that grid** before JPEG encoding. Both arms optimized the maximum SFace/GhostFaceNet normalized source-match margin, `(cosine − threshold)/(1 − threshold)`, against the same untransformed clean neutral-front embeddings. The four objective conditions were export, JPEG75/4:2:0, blur and crop90. JPEG, rounding, blur and detection have no derivative in the H7 fixed-affine backward approximation; the Ghost gradient used the parity-checked float32 clone of the native author H5.

The run made **three forward/gradient evaluations but only two updates reflected in the selected JPEG**. At each step it generated the forward JPEG, calculated the gradient and then updated Adam; the step-3 **pre-update forward JPEG** was frozen unconditionally. Thus each arm used 24 edited gradient model-condition forwards (3 steps × 4 conditions × 2 models) and one step-3 native diagnostic with eight model-condition queries. Native scores did not choose an image. All four selected JPEGs and their hashes were frozen before any smiling or three-quarter same-person view was opened. The one-photo optimizer used only neutral-front source images for `frll-001` and `frll-003`.

The geometric dot area is `π(0.3)² = 0.282743` of a full cell; the inverse covers the rest **within** the 14×14 grid. A model-free synthetic check confirmed zero overlap, exact in-grid union and zero rendered pixels outside the grid. Before any arm search, the run recorded both clean-box, face-masked pixel counts:

| Identity | In-grid face pixels | Dot pixels / fraction | Inverse pixels / fraction |
| --- | ---: | ---: | ---: |
| 001 | 204,653 | 58,222 / 28.449% | 146,431 / 71.551% |
| 003 | 187,567 | 53,372 / 28.455% | 134,195 / 71.545% |

The primary gate required root appearance screening to pass, all seven processed conditions to detect the selected face, and inverse dots to reduce **each person's** ArcFace worst-condition own-gallery cosine by at least `0.05` versus dots while worsening each SFace and GhostFaceNet worst normalized margin by no more than `0.02`. Within-person processed face-RMS differences above `0.25` would flag an unbalanced comparison. ArcFace was the third **development** model, excluded from the gradient objective but already informing this research. It was neither a final-panel model nor an independent holdout.

| Identity | Arm | ArcFace worst gallery cosine | SFace worst gallery cosine | GhostFaceNet worst gallery cosine |
| --- | --- | ---: | ---: | ---: |
| 001 | dots | 0.556686 | 0.543660 | 0.426526 |
| 001 | inverse dots | 0.617093 | 0.617095 | 0.560878 |
| 003 | dots | 0.603987 | 0.562727 | 0.549633 |
| 003 | inverse dots | 0.596368 | 0.629743 | 0.500366 |

Both clean controls were eligible on all three development models. All **84 edited model-condition evaluations** (four JPEGs × seven conditions × three models) were valid; there were no final detection/selection failures. ArcFace matched the own four-photo gallery in **all 28 edited conditions** at threshold `0.2376004863`. Its worst-gallery reduction (`dots − inverse`) was `−0.060407` for 001 and `+0.007619` for 003, below `+0.05` for each. Normalized worst-margin worsening (`inverse − dots`) was `+0.151425` SFace and `+0.204634` GhostFaceNet for 001, and `+0.138188` SFace and `−0.075040` GhostFaceNet for 003. Three of the four native comparisons exceeded the allowed `+0.02`. Neither arm passed all seven conditions on either native model. The predeclared gate failed.

Pre-JPEG RMS was `16.0003–16.0016`; exact export face RMS was `16.0479–16.0576`. The largest same-person, same-condition arm RMS gap was `0.140` (blur), below the `0.25` imbalance flag. Post-export **changed-face coverage nevertheless differed**: dots changed `51.6–52.0%` of face-mask pixels and inverse dots `86.2–86.8%`, including JPEG effects. Equal face RMS therefore does not isolate support from local amplitude/energy distribution. The four searches took `6.01–6.21` seconds each, `27.30` seconds together excluding model loading; separate ArcFace scoring took `24.31` seconds. These are native CPU research timings, not a browser runtime.

Root reviewed full photos and face crops before ArcFace scoring: the square/circular-window pattern looked conspicuously artificial, with plausible facial structure and no anatomy-like artifacts. This is root screening, **not independent human identity acceptance**.

The private external directories `runs/support-probe-v1` and `runs/support-probe-v1-arcface` contain exact selected and processed JPEGs, source snapshots, model copies, pre-optimization support counts, one diagnostic per arm, all four-reference scores, distortion metrics and full/face contact sheets. Nothing biometric is distributed in Git. SHA-256 fingerprints:

| Artifact | SHA-256 |
| --- | --- |
| Executed `optimize_support_probe.py` | `59e5954a38d548f8f8a7434532ec825b9ebe572bfd6cad92899feadfba49b4da` |
| Frozen protocol | `648d31ae26ded4f44630a96e55ff71125be62c4b2f673037e1da621793140fb7` |
| Pre-optimization support counts | `8acbbd3c132ffb6fb5cbf3372e9a7036da692531a389733fb84ad858700619bb` |
| Four-JPEG frozen manifest | `22de285490d4af3e91e9a833258b99b45c2b34870f87320a53917ea83002e9dd` |
| Native seven-condition results | `3f79d0945fd210abe9a977f61940cde620719b25bf1cc113089fbea9aa40a9cc` |
| ArcFace seven-condition results | `c31565f1aede38168a72ea767e53e52f0c9d95dc4d7af446c5d36c157143a11c` |
| Distortion comparison | `c31afe68adb13b9b9eb964b4ac0b803723ef421d68644fa71d54e1a49db3191a` |

The protocol records the FRLL manifest, native calibration and model hashes, previous Ghost float32-clone diagnostic and imported H10 surrogate hash. The runner validates these inputs, checks private model copies before loading, and writes exact source snapshots. Reproduce only with permitted local FRLL data, official calibrated weights and fresh **external** output paths:

```powershell
$py = '<PYTHON_WITH_RESEARCH_DEPENDENCIES>'
$data = '<EXTERNAL_FCKFACE_DATA_ROOT>'
$weights = '<EXTERNAL_CALIBRATED_MODEL_DIRECTORY>'
& $py research/optimize_support_probe.py `
  --manifest "$data/frll/manifest.json" `
  --sface-calibration "$data/runs/calibration-sface-v1/calibration.json" `
  --ghostface-calibration "$data/runs/calibration-ghostface-v1/calibration.json" `
  --yunet "$weights/face_detection_yunet_2023mar.onnx" `
  --sface "$weights/face_recognition_sface_2021dec.onnx" `
  --ghostface "$weights/ghostfacenet_v1.h5" `
  --clone-diagnostic "$data/runs/ensemble-dots-v1/ghost-float32-clone-diagnostic.json" `
  --out "$data/runs/support-probe-v1-REPRO"

& $py research/score_frozen_arcface.py `
  --candidates "$data/runs/support-probe-v1-REPRO/frozen-inputs.json" `
  --manifest "$data/frll/manifest.json" `
  --calibration "$data/runs/arcface-development-v1/calibration.json" `
  --output "$data/runs/support-probe-v1-arcface-REPRO"
```

The first command freezes four outputs before native gallery scoring. The second scores only those frozen JPEGs on the development ArcFace pipeline. Different data/source/model hashes constitute a new run. The four reserved final recognizers were not used.
