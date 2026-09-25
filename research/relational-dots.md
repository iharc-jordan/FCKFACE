# H8B relational dots: bounded development result

H8B tested whether coordinating changes in two face models' **identity-relation vectors** improved transfer from a single source photo. It did not at the frozen settings. The relational arm increased, rather than decreased, the worst ArcFace own-gallery cosine for both pilot identities relative to both controls. This is a two-person development result, not an independent privacy or product validation. The H8B setting is retired without expansion or reseeding. Inter-sample relations already appear in face-model work such as [CoupleFace](https://arxiv.org/abs/2204.05502); this unsuccessful experiment does not establish novelty.

The renderer is H7's fixed-condition 14×14 circular RGB dot carrier, with face RMS target 16, per-channel cap 64, fixed seed 0, Adam learning rate 0.15, and 18 steps. Its four gradient conditions are export, JPEG 75/4:2:0, blur, and 90% crop. There are 144 differentiable model-condition forwards per arm (18 steps × 4 conditions × 2 models); six arms made 864 such forwards. JPEG, blur, rounding, and alignment-grid gradients use the H7 fixed-affine BPDA approximation. SFace uses converted ONNX for its gradient, and GhostFaceNet uses the parity-checked float32 clone of the author H5; native calibrated models supply checkpoint diagnostics. The surrogate is approximate to the native author model.

For each native model, an anchor is the unit-normalized centroid of **four unit clean embeddings** from one other development identity. The pool is fixed by four-view validity on *both* models before any attack: 52/52 eligible development identities. The eight reserved H8 evaluation IDs (021, 026, 029, 030, 032, 037, 041, 042) and all calibration/held-out identities are excluded. The ordered anchor arrays are stored outside Git.

For a unit embedding `z`, let `r_m(z) = normalize(center([z · a_m,i]_i))`. For the neutral-front source relation `r_m,0`, the edited displacement is `v_m = r_m,edit − (r_m,0 · r_m,edit) r_m,0`; **v is not normalized**. The relational agreement is `A = min_condition(v_SFace · v_Ghost)`. All arms minimize the maximum normalized source-match margin over the two models and four conditions. The relational arm additionally subtracts `0.25 A`. The baseline uses only the margin. The correspondence control uses the same relational loss after a fixed seed-20260925 permutation of Ghost anchors, applied consistently to its source and edited relation; it keeps the same renderer and cost. Both framework partial gradients of the active agreement and margin are propagated to the shared dots. Synthetic central differences for aligned and permuted correspondences differed from the Torch and TensorFlow partials by at most `2.1e-10` and `5.1e-11`, respectively; zero displacement gave zero agreement.

**Selection was changed before H8B gallery scoring:** every arm unconditionally exports the forward JPEG at step 18. Native checks at steps 3, 6, 9, 12, 15, and 18 are diagnostic only. There is no best-native-score choice, fallback, strength adjustment, or seed search. All six JPEGs for frll-021 and frll-026 were frozen and hashed before any of their additional same-person gallery views were read. Invalid detector results remain inconclusive. Seven final conditions and four separately captured own-gallery views were then evaluated on development SFace, author GhostFaceNet, and frozen ArcFace. The ArcFace criterion was declared before scoring: relational must reduce the median across the two people of the **worst-condition own-gallery cosine** by at least 0.02 versus **each** control and improve **both** individuals with valid conditions. Lower cosine is better for this test.

| Identity | Arm | ArcFace worst gallery cosine | ArcFace valid / 7 | Native SFace valid / 7 | Native Ghost valid / 7 | Export face RMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| frll-021 | baseline | 0.592979 | 7 | 0 | 0 | 16.036 |
| frll-021 | relational | 0.609110 | 7 | 0 | 0 | 16.066 |
| frll-021 | permuted | 0.603247 | 7 | 4 | 4 | 16.047 |
| frll-026 | baseline | 0.628492 | 7 | 7 | 7 | 16.063 |
| frll-026 | relational | 0.636106 | 7 | 7 | 7 | 16.073 |
| frll-026 | permuted | 0.630162 | 7 | 7 | 7 | 16.051 |

Both clean identities were eligible on all three calibrated models. ArcFace matched own identity in **all 42 candidate conditions** (0/42 nonmatches; threshold 0.2376004863). The relational median gain, defined as control worst cosine minus relational worst cosine, was **−0.011873** versus baseline and **−0.005904** versus permuted correspondence; it was negative for each person against each control. The specified +0.02 gain and both-person improvement therefore failed. On frll-021, SFace and Ghost did not detect any of the seven baseline or relational conditions; the permuted arm was valid in four. Those failures are inconclusive, not nonmatches. All frll-026 arms were valid on both native models but matched own identity in at least one condition. The export face RMS range was 16.036–16.073; within-person arm spreads were 0.030 and 0.022. Export changed-face fractions were approximately 0.496–0.498 and 0.519–0.522. The small outside-face differences measured after export arise from JPEG encoding, not a claim of pixel-exact preservation in the JPEG.

The six optimizations took 328.4 seconds after anchor construction (individual arm times 49.2–59.6 seconds). The anchor pass made 52 × 4 × 2 = 416 native model-view evaluations. Six native checkpoint passes made 6 × 6 × 4 × 2 = 288 model-condition evaluations, without selecting an image. The ArcFace scorer recorded 33.6 seconds summed across six candidate scoring records, excluding startup. Root visually reviewed full-photo and face-crop sheets: the dots read as artificial artwork and facial geometry remained plausible; that is **root screening only**, not independent human confirmation of same identity or appearance acceptance.

H8A, the earlier native-checkpoint-selected variant, is retained separately and **has no interpretable attack outcome**: its frll-021 baseline found no valid checkpoint and its first execution failed to persist the per-checkpoint rows. The overlapping conversion workflow was subsequently found to have changed the shared SFace ONNX from calibrated SHA-256 `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` to `827d2b58fe491fa70fe96d1b4998a07ac0c3e58a0e81033a0ffb713793c3a0e9`. Intermediate model-load/hash timing was not recorded, so the original H8A anchor and checkpoint evidence may be contaminated even if the graph change was semantically equivalent. A separately labelled baseline-only replay with restored, private hash-verified model copies saved all six checkpoint JPEGs: both native models reported `no_face` in all four objective conditions at every checkpoint. That replay supports an edited-detection diagnosis for this baseline, but does not retroactively validate the original execution. H8B regenerated all 52 anchors from run-local exact-byte model copies and changed only the declared fixed-step selection before reading a gallery.

No H8 portraits, embeddings, anchors, model weights, or experiment outputs are in the source checkout. External run directories are `runs/relational-dots-v1` (H8A original), `runs/relational-dots-v1-diagnostic-replay`, `runs/relational-dots-v2-fixed-step`, and `runs/relational-dots-v2-arcface` beneath the chosen private FCKFACE data root. The H8B directory includes executed-source snapshots, `protocol.json`, native checkpoint JPEGs and reasons, `anchors.npz`, `frozen-inputs.json`, six selected JPEGs, seven-condition JPEGs and face crops, per-reference scores, `distortion.json`, and labeled contact sheets. Key SHA-256 fingerprints:

| Artifact | SHA-256 |
| --- | --- |
| H8B executed `optimize_relational_dots.py` | `e42c85984c5b5c882e53c7fb6c02d41294bf9090ae35041aa6131db6b073f37f` |
| H8B protocol | `768b6e983fb20a1d04daa09b280528397e85f1e43d4b356278549f57b38c1cd0` |
| H8B freeze results | `313dd39c7aab908077b348f10c81222a327bccac01776544e9a401b318489dc8` |
| Regenerated anchors | `87906405f4957361a6d52725327cb77d4f71db4819f020165edb77781ff366f9` |
| Frozen six-JPEG input manifest | `b08143e12268b441ceb04b56702512bf26e828ee878a17fcdf3db6c5ddc4a2f6` |
| H8B native seven-condition results | `cee94666ccf6a9a4a25a717120635cc6497144431f54c9ca683d248db32989af` |
| ArcFace seven-condition results | `3a51208d71ea943909ffe33e7aac5a46344f6370828042a7f06f23af9e57ab57` |
| Distortion comparison | `a64ca1aca66b192feb1cb0bcd3fae0297ad6e9b2c3e2f5a6aad5c128b0093b7f` |
| H8A original executed source | `0f6a60a3d206464ae289a20a5e7d265cd5d835cf330cce04c857634cc4d02126` |
| Calibrated YuNet | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| Calibrated SFace | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |
| Calibrated author GhostFaceNet H5 | `e4182ca2470dac3eb79974f4d5d54f2abdf66d7e231c6f3df4059a90bded1271` |
| ArcFace calibration | `a186f3d285f4cff65d8c187a65f5929c19b13b86f8170583dee34dccab9bb56a` |

The H8B private run records include every selected JPEG hash, the FRLL development manifest (`2509c48bd7852251f7a3778cc7e1e320e17003d1bb56cca6dbf84998b4931600`), SFace and Ghost calibrations, and the exact model inputs. Reproduction requires the permitted FRLL research data, official calibrated models, the prior float32-clone diagnostic, and a fresh external output directory. Replace the placeholders in these PowerShell commands with local paths; they are **not** public artifact locations:

```powershell
$py = '<PYTHON_WITH_RESEARCH_DEPENDENCIES>'
$data = '<EXTERNAL_FCKFACE_DATA_ROOT>'
$weights = '<EXTERNAL_CALIBRATED_MODEL_DIRECTORY>'
& $py research/optimize_relational_dots.py `
  --manifest "$data/frll/manifest.json" `
  --sface-calibration "$data/runs/calibration-sface-v1/calibration.json" `
  --ghostface-calibration "$data/runs/calibration-ghostface-v1/calibration.json" `
  --yunet "$weights/face_detection_yunet_2023mar.onnx" `
  --sface "$weights/face_recognition_sface_2021dec.onnx" `
  --ghostface "$weights/ghostfacenet_v1.h5" `
  --clone-diagnostic "$data/runs/ensemble-dots-v1/ghost-float32-clone-diagnostic.json" `
  --out "$data/runs/relational-dots-v2-fixed-step-REPRO"

& $py research/score_frozen_arcface.py `
  --candidates "$data/runs/relational-dots-v2-fixed-step-REPRO/frozen-inputs.json" `
  --manifest "$data/frll/manifest.json" `
  --calibration "$data/runs/arcface-development-v1/calibration.json" `
  --output "$data/runs/relational-dots-v2-arcface-REPRO"
```

The first command regenerates anchors, optimizes and freezes both identities' three arms, then scores native SFace/Ghost after the complete freeze. The second command scores only the already-frozen JPEGs on ArcFace. A reproduction with different source/model hashes is a new run, not a verification of these outcomes.
