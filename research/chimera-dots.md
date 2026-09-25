# H13 chimera-aware dot optimization: development screen

**The fixed two-person setting failed its predeclared gate and was retired without expansion.** Of 84 edited model-condition evaluations, 40 were valid and 44 had no detected face. ArcFace validated all 28 queries but matched the person in every condition. A missed detection is inconclusive, never evidence of protection.

## Frozen comparison

H12's eye-positive blue chimera was rebuilt from the original `neutral_front` photograph and its frozen clean YuNet landmarks. Exporting that pre-JPEG reconstruction reproduced each H12 base JPEG byte-for-byte. H13 placed H5's literal 14×14 radius-0.3 dots only where they intersected the H12 hard treatment support. The original RGB eye band and pixels outside that support stayed unchanged before JPEG. Both arms used 588 RGB coefficients initialized from seed 0, a per-channel cap of 64, and the **same feasible additive field**. Its bounds were `low=max(-64,-original,-chimera)` and `high=min(64,255-original,255-chimera)` per channel, preventing substrate-dependent clipping.

The *stack* arm optimized SFace/GhostFaceNet source-only loss on `original + field`; the *joint* arm optimized the same loss on `chimera + field`. **Both final exports were `chimera + field`.** Thus the comparison changes the optimization substrate under a common final-image feasible set. It does not isolate contrast polarity itself. The loss used H7's fixed clean landmarks per condition and the maximum normalized matching margin across SFace/GhostFaceNet and export, JPEG75, blur and crop90. No other same-person reference image entered optimization.

At each gradient step, the shared detached scalar projector targeted `sqrt(pre-JPEG H12 base face RMS² + 16²)` against the original RGB source, with the face-RMS support used by `actual_distortion`. It geometrically bracketed the first **observed** tolerance crossing and bisected the bounded field. A finite search cannot certify a global first crossing for a nonmonotone function. The exact-JPEG-per-step prototype cost about 3.1–3.3 seconds per projection, so the predeclared protocol instead used the fast pre-JPEG projector during optimization and **one exact Q95 JPEG scalar correction** of the fixed step-18 coefficients before export. That correction targeted `sqrt(H12 decoded-base-versus-clean-JPEG face RMS² + 16²)` within 0.02, using no model scores or alternate coefficients. An unreachable target would invalidate the arm; none was unreachable here. The correction changed only scalar magnitude, not coefficient direction, mask, base, palette or geometry.

Each arm made 18 forward/gradient steps with Adam learning rate 0.15: the unconditionally frozen step-18 forward JPEG reflected **17 effective updates**, while all 18 steps contributed to the declared 144 edited model-condition gradient forwards per arm. Every three steps, the runner logged surrogate source values and projection metadata; it made **zero native checkpoint queries** and selected no checkpoint. All four JPEGs were frozen before any separate same-person gallery view was read. Root then reviewed both full-photo and face-crop sheets and found visibly artificial blue fields and regular dots with plausible facial structure and original eyes, without vein-like or anatomical artifacts. This preliminary screen is not independent human same-person acceptance.

The seven-condition development scoring protocol, root-screen record and four JPEG hashes were frozen **before** gallery/model access. Existing calibrated SFace, author GhostFaceNet and official SCRFD/ArcFace scorers used the clean source plus three separately captured own-identity references. Eligibility required four valid clean references and matched clean controls in all seven conditions. All 12 arm/model clean controls were eligible. OpenVINO 0095 and all four reserved final recognizers were excluded. ArcFace is a development model that had already informed earlier research, not an independent holdout.

## Result and falsification

Worst gallery cosine is the maximum over valid conditions and four own-identity references; lower is better. `valid/NM` counts valid edited conditions and nonmatching valid conditions. A dash means no valid query, not a nonmatch. Thresholds were SFace 0.515038, GhostFaceNet 0.343449 and ArcFace 0.237600.

| Identity / arm | SFace worst; valid/NM | Ghost worst; valid/NM | ArcFace worst; valid/NM |
| --- | ---: | ---: | ---: |
| 032 / stack | —; 0/0 | —; 0/0 | 0.552359; 7/0 |
| 032 / joint | —; 0/0 | —; 0/0 | 0.490889; 7/0 |
| 037 / stack | —; 0/0 | —; 0/0 | 0.526907; 7/0 |
| 037 / joint | 0.640661; 6/0 | 0.354227; 6/2 | 0.583239; 7/0 |

All 44 inconclusives were `probe_no_face`: all seven conditions of both 032 arms and the 037 stack arm on both native models, plus crop90 of 037 joint on each native model. GhostFaceNet's 037 joint had two nonmatches among six valid conditions, but the other four matched and crop90 was inconclusive. ArcFace matched **28/28** valid edited conditions. Relative to the stack arm, joint reduced ArcFace worst cosine by 0.061469 for 032 but **increased** it by 0.056332 for 037. Relative to the unchanged H12 chimera (032: 0.494497; 037: 0.504418), joint reduced it only 0.003608 for 032 and increased it 0.078821 for 037. The predeclared requirement was a reduction of at least 0.05 versus **both** controls on **each** identity, with both arms valid in all seven conditions and joint nonmatching every condition on both native models. It failed each of those recognition requirements.

The final export face RMS was approximately 59.444 for 032 and 63.974 for 037. The largest within-person, same-condition stack/joint RMS difference across all seven processed conditions was 0.02922 for 032 and 0.04834 for 037, inside the 0.25 imbalance gate. Scalar-correction residuals were about −0.019 RMS. The four optimizations took 31.08–33.85 seconds each, excluding model loading; this is not a browser runtime measurement. The experiment retires this fixed H13 setting, not every combined graphic/optimization method. No novelty, privacy, release or independent-transfer claim follows.

## Private evidence and reproduction

Photos, embeddings, model weights, exact exported/processed JPEGs and scores remain outside Git under private `FCKFACE-data/runs/chimera-dots-v1` and `chimera-dots-v1-development-screen`. The executed source and model copies were archived before optimization. SHA-256 fingerprints:

| Artifact | SHA-256 |
| --- | --- |
| Executed H13 source | `05d33fbbb90e2c499b83903753d4209bda29c6f0f2bd1aef15c481b2b27e881a` |
| Optimization protocol / four-export freeze | `64682c04e154536d908cab051699dfa947deadab9036369ec7d34bd48f1b14da` / `95e4b04bfecf38d4d7c43f20d42ce5b9129d062f4ed4591b5989c40dc0aab0b7` |
| Four-JPEG manifest / root appearance record / scoring protocol | `9f384685223c4e8ef27cbac995446c756aaa7886fbe46ab8c3d9da3aedbed363` / `ce89c59266d0cc497655c6eefc9299055056c1d1428414f1b1e95d2781f2339e` / `11ac02989fabe54774f63ff5f45f8779f3748cd276971ab248017a67dbe4c91d` |
| SFace / GhostFaceNet / ArcFace result JSON | `0c90e619efe1387021a0a1c13bed843ec80a832186681b34df7f60dfca1bb58b` / `b1009027784ca49915ba837e148c2585f5ee174accb895240cb8d32b7b0692ae` / `5096e90727b5602116ea913897f45e52d0d9bd7d52c4173f77f108a404fd2f5b` |
| Audited disposition JSON | `091e31a7b5167362bedcfbd39b2e86ff2dcbe6ee9c060790d8d7d40a59626956` |

The four JPEG hashes, per-reference scores, invalid reasons, projection scales and exact per-condition distortions are in those private records. To reproduce in **new** external directories with the pinned dependencies and calibrated artifacts, use:

```powershell
$py = '<PYTHON_WITH_RESEARCH_DEPENDENCIES>'
$data = '<EXTERNAL_FCKFACE_DATA_ROOT>'
$weights = '<EXTERNAL_CALIBRATED_MODEL_DIRECTORY>'
$registry = '<EXACT_HISTORICAL_CALIBRATED_SFACE_REGISTRY_JSON>'
& $py research/optimize_chimera_dots.py `
  --manifest "$data/frll/manifest.json" `
  --h12-frozen "$data/runs/contrast-chimera-v1/frozen-appearance.json" `
  --sface-calibration "$data/runs/calibration-sface-v1/calibration.json" `
  --ghostface-calibration "$data/runs/calibration-ghostface-v1/calibration.json" `
  --yunet "$weights/face_detection_yunet_2023mar.onnx" `
  --sface "$weights/face_recognition_sface_2021dec.onnx" `
  --ghostface "$weights/ghostfacenet_v1.h5" `
  --clone-diagnostic "$data/runs/ensemble-dots-v1/ghost-float32-clone-diagnostic.json" `
  --out "$data/runs/chimera-dots-REPRO"

$frozen = "$data/runs/chimera-dots-REPRO/frozen-inputs.json"
& $py research/score_frozen_sface.py --candidates $frozen --manifest "$data/frll/manifest.json" --calibration "$data/runs/calibration-sface-v1/calibration.json" --models-registry $registry --yunet "$weights/face_detection_yunet_2023mar.onnx" --sface "$weights/face_recognition_sface_2021dec.onnx" --output "$data/runs/chimera-dots-sface-REPRO"
& $py research/score_frozen_ghostface.py --candidates $frozen --manifest "$data/frll/manifest.json" --calibration "$data/runs/calibration-ghostface-v1/calibration.json" --yunet "$weights/face_detection_yunet_2023mar.onnx" --ghostface "$weights/ghostfacenet_v1.h5" --output "$data/runs/chimera-dots-ghostface-REPRO"
& $py research/score_frozen_arcface.py --candidates $frozen --manifest "$data/frll/manifest.json" --calibration "$data/runs/arcface-development-v1/calibration.json" --output "$data/runs/chimera-dots-arcface-REPRO"
```

These commands describe reproduction; they did not run a second experiment. The frozen H12 appearance manifest contains absolute paths to its saved base JPEGs. The executed H13 runner verifies those paths and bytes, so relocation to a different filesystem layout requires the same private H12 artifact paths; the placeholders alone do not make the frozen input manifest portable.
