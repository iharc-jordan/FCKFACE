# H10 optimizer and input-diversity dots: bounded development result

H10 asked whether momentum and stochastic input diversity would transfer a fixed regular-dot edit to a third development recognizer, excluded from the optimization gradients, better than Adam. ArcFace had already informed earlier development; it was not an independent holdout. At the predeclared two-person setting, **MI+DI was worse than both controls on both people**, and every edited ArcFace condition still matched its own four-photo gallery. Retire these settings without expansion or reseeding. This is a development result, not a privacy or product validation. Momentum iterative attacks and diverse input transforms are established methods: [Dong et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.html) and [Xie et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Xie_Improving_Transferability_of_Adversarial_Examples_With_Input_Diversity_CVPR_2019_paper.html). H10 makes no novelty claim for either component.

## Frozen comparison

The two development identities were `frll-029` and `frll-030`, with only each `neutral_front` image available to optimization. Every arm used the same H7 fixed-condition 14×14 coloured-dot carrier, seed 0 initialization, face RMS target 16, per-channel cap 64, 18 steps and four objective conditions: exported JPEG, JPEG75/4:2:0, blur and 90% crop. The loss was the maximum over two model-specific normalized source-match margins, `(cosine − calibrated_threshold)/(1 − calibrated_threshold)`. **Every arm compared its edited crops against the same, untransformed native clean neutral-front embedding for each model.** The other three same-person views were not read until all six outputs were frozen.

| Arm | Coefficient update | Edited-crop input diversity |
| --- | --- | --- |
| Adam | Adam, learning rate 0.15 | None |
| MI | Momentum 1.0; divide coefficient gradient by its mean absolute value, then sign-descent step 0.15 | None |
| MI+DI | Same MI update | Frozen seed-20260925 schedule: per step and condition, probability 0.5 identity; otherwise bilinear resize to an integer side 101–111 and randomly place in a zero-RGB 112×112 frame |

The DI schedule was generated before optimization and shared by the SFace and GhostFaceNet edited crops; it contained 39 identity and 33 resized/padded choices among 72 step-condition entries. DI did **not** transform or recache the clean reference. The resize/pad Jacobian was included for both gradients, including the TensorFlow GhostFaceNet crop gradient injected into the Torch carrier graph. A synthetic cross-framework directional finite difference differed by `3.87e−10`; it tests that handoff, not face-model parity. H7's compiled Ghost gradient was required to pass its eager/compiled check. Its parity-checked float32 clone supplied gradients; the original author H5 and native OpenCV SFace supplied checkpoint and final measurements. JPEG, rounding, blur and detector coordinates still use H7's fixed-affine approximate backward pass.

Each arm made exactly 144 edited surrogate model-condition forwards (18 × 4 × 2), and 48 native checkpoint model-condition queries (six checkpoints × 4 × 2). Checkpoints were diagnostic only: the **forward JPEG at step 18** was selected unconditionally, before that step's coefficient update. No best-checkpoint choice or fallback was made. Six selected JPEG hashes were written to `frozen-inputs.json` before opening additional same-person views. Seven final processing conditions and four clean gallery views per person were then scored by native SFace and GhostFaceNet, followed by the separately frozen official ArcFace development scorer. Invalid detection or selection would be inconclusive, never a nonmatch.

The predeclared transfer criterion required MI+DI to reduce the **median across the two people** of the worst-condition ArcFace own-gallery cosine by at least `0.02` versus **each** control, with a positive reduction on **both** people and all seven conditions valid. Lower cosine is better. The ArcFace threshold was frozen at `0.2376004863`; SFace and GhostFaceNet used `0.5150383510` and `0.3434492487` respectively.

## Result

Both clean people were eligible on all three models. All 126 edited model-condition evaluations (six JPEGs × seven conditions × three models) were valid; there were **no detector or selection failures** in the final evaluation. ArcFace matched the own gallery in **all 42 candidate conditions**.

| Identity | Arm | ArcFace worst gallery cosine | SFace worst gallery cosine | GhostFaceNet worst gallery cosine | Native all-condition nonmatch? |
| --- | --- | ---: | ---: | ---: | --- |
| 029 | Adam | 0.448252 | 0.343597 | 0.198560 | Both models |
| 029 | MI | 0.502479 | 0.402082 | 0.251638 | Both models |
| 029 | MI+DI | 0.507174 | 0.444092 | 0.241810 | Both models |
| 030 | Adam | 0.588494 | 0.574857 | 0.408429 | Neither |
| 030 | MI | 0.605418 | 0.586024 | 0.390803 | Neither |
| 030 | MI+DI | 0.612109 | 0.576557 | 0.412027 | Neither |

The ArcFace reduction (`control − MI+DI`) was `−0.058922`/`−0.023615` against Adam for 029/030, median **`−0.041268`**, and `−0.004695`/`−0.006691` against MI, median **`−0.005693`**. Both controls beat MI+DI for both people. The `+0.02` criterion failed decisively. On the native models, all 029 arms were valid nonmatches in every condition; 030 retained at least one own-gallery match in every arm on both models. This native partial effect does not override the failed third-model result.

Pre-JPEG face RMS was `16.001–16.002`. Exact exported face RMS was `16.050–16.075` across the six arms; subsequent condition RMS ranged `15.168–16.143`. The largest within-person, same-condition arm spread was `0.078` RMS. Export changed-face fractions were about `0.480–0.507`. These are closely distortion-matched comparisons, not pixel-identical edits or exact equality after processing. Search time was `38.0–40.4` seconds per arm and `240.1` seconds for all six after model loading; the separate ArcFace scorer recorded `31.4` seconds. Peak observed H10 process working set was about `1.41 GiB` with at least about `5.8 GiB` system RAM free during checks. This is native CPU timing, not browser runtime.

Root reviewed the full photos and face crops: the coloured circles read as artificial and facial structure appeared plausible. This is **root screening only**, not independent human confirmation that every edited portrait unequivocally depicts the same person.

## Reproduction and provenance

The run-local external directories `runs/transfer-dots-v1` and `runs/transfer-dots-v1-arcface` contain the exact JPEGs, source snapshot, private model input copies, protocol, frozen DI schedule, six diagnostic checkpoint JPEGs per arm, native and ArcFace seven-condition JPEGs, face crops, four-reference scores, distortion metrics, hashes, timings and full/face contact sheets. They are not in Git. Key SHA-256 fingerprints:

| Artifact | SHA-256 |
| --- | --- |
| Executed `optimize_transfer_dots.py` | `a04a0d6b42fb8f6375cca20bcc871ff46cf659049c5ceb5977d1f345111eb273` |
| Protocol | `556657ca18028552980055973914024aebda1d0d11d194ccc37927b99b3abb19` |
| Frozen DI schedule | `f7b5aea97a151a81e724b74c6052f98d3d005d2615baef3227ab6aef7b248a87` |
| Six-JPEG frozen manifest | `e31005186b4e0a291665e4c1418d341af21e811093222bf41d85917d30be9017` |
| Native seven-condition results | `c068309288696f14dd3247ba9bf267394fb7318b0c79f7c5603eab32c4956b74` |
| ArcFace seven-condition results | `b182b324c338044398dd5ddb7fc5f82c25df4e3410a4d26e5db055209e333434` |
| Distortion metrics | `b45c9d7c56c8213e811f00a88af8ceaf4c3ed16908a32c640906633c416c3193` |

The protocol pins the FRLL manifest (`2509c48bd7852251f7a3778cc7e1e320e17003d1bb56cca6dbf84998b4931600`), both native calibration files, the previous float32-clone diagnostic, and model artifact hashes. The runner checks those hashes, copies calibrated models to the private output directory, and checks the copies before loading them. Use permitted local FRLL data and official calibrated weights; substitute private paths in these PowerShell commands and choose **new external** output directories:

```powershell
$py = '<PYTHON_WITH_RESEARCH_DEPENDENCIES>'
$data = '<EXTERNAL_FCKFACE_DATA_ROOT>'
$weights = '<EXTERNAL_CALIBRATED_MODEL_DIRECTORY>'
& $py research/optimize_transfer_dots.py `
  --manifest "$data/frll/manifest.json" `
  --sface-calibration "$data/runs/calibration-sface-v1/calibration.json" `
  --ghostface-calibration "$data/runs/calibration-ghostface-v1/calibration.json" `
  --yunet "$weights/face_detection_yunet_2023mar.onnx" `
  --sface "$weights/face_recognition_sface_2021dec.onnx" `
  --ghostface "$weights/ghostfacenet_v1.h5" `
  --clone-diagnostic "$data/runs/ensemble-dots-v1/ghost-float32-clone-diagnostic.json" `
  --out "$data/runs/transfer-dots-v1-REPRO"

& $py research/score_frozen_arcface.py `
  --candidates "$data/runs/transfer-dots-v1-REPRO/frozen-inputs.json" `
  --manifest "$data/frll/manifest.json" `
  --calibration "$data/runs/arcface-development-v1/calibration.json" `
  --output "$data/runs/transfer-dots-v1-arcface-REPRO"
```

The first command freezes six outputs before the native gallery pass; the second scores only those frozen inputs. Different source, model or data hashes define a new run rather than a verification of these results. No reserved final recognizer was used.
