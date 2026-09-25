# H12 contrast chimera: appearance-first development screen

**The fixed setting failed its predeclared development gate and is retired without expansion.** All 63 edited model-condition evaluations were valid, but both chimeras matched on SFace and official ArcFace in every condition. This is a development result, not evidence of anonymity or independent transfer.

## Appearance hypothesis and frozen render

[Gilad, Meng and Sinha (PNAS 2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2664053/) found that keeping the eye region in positive contrast improved **human** recognition of otherwise negative faces relative to full negatives. That motivated an artificial portrait treatment intended to retain legible eyes and facial structure. It does not predict machine-recognizer failure, and contrast chimeras, photo negatives and solarization are prior art. H12 makes no novelty claim for the graphic transformation.

Only the `neutral_front` source images for development identities `frll-032` and `frll-037` were read for rendering. The originals were preserved. Clean YuNet selected one face and its eyes; no gallery view, recognizer, embedding, or reserved final model was used during rendering. Let `y=(0.2126R+0.7152G+0.0722B)/255` on decoded sRGB and define the fixed blue palette `P(t)=(31,45,111)+t(108,168,120)`, rounded to output RGB. Inside the existing oval face mask (`face_mask >= 0.5`), leave the original eye/eyebrow band unchanged: `|u| <= 0.95` and `-0.35 <= v <= 0.30`, where `u,v` are coordinates along/across the eye axis in interocular-distance units. The chimera applies `P(1-y)` **outside** that band. Its treated core has a genuine negative luminance slope; there is no blend, drawn contour or geometry change. Pixels outside the mask remain source pixels. The monotone control uses identical geometric support and `P(clamp(y+b,0,1))`, which has nondecreasing slope. A full-negative diagnostic uses `P(1-y)` across the entire face mask, including the eyes.

The fixed palette, band, three arms and two people were declared before six JPEGs were rendered. For each monotone control, `b` was chosen using only decoded Q95/4:4:4 JPEG face RMS against the correspondingly exported clean source: scan nearest zero first in `[-1,0]` at 1/64 intervals, then `[0,1]` only if needed, followed by 18 bisections. The prespecified balance tolerances were `0.25` face RMS and `0.02` changed-face fraction; lack of a bracket or imbalance would be retained, not tuned using recognition. This experiment deliberately abandoned the earlier RMS-16 dot budget.

| Identity | Chimera / control face RMS | Changed-face fraction | Control `b` | Control clipped treated fraction | Visual decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 032 | 57.270428 / 57.270409 | 0.737732 / 0.738219 | −0.265659 | 0.007516 | Both approved for development scoring |
| 037 | 61.961716 / 61.961700 | 0.730935 / 0.730835 | −0.474592 | 0.553704 | Chimera approved; control rejected |

The RMS and coverage were closely matched for both pairs, but the 037 control clipped **55.37% of treated pixels**, erasing substantial facial detail. Root reviewed the full photos and face crops at 18:49 UTC **before any H12 recognition score**. Both chimeras and the 032 monotone control retained plausible facial structure and had an unmistakable graphic blue treatment. Both full negatives were rejected because the eyes and likeness were uncertain; the 037 monotone control was rejected for detail erasure and uncertain likeness. All six outputs remain frozen outside Git, but the three rejected JPEGs were **never scored**. This is root screening, not independent human same-person acceptance. Because the 037 matched control failed appearance review, there is no isolated polarity comparison for 037. Even the numerically balanced 032 comparison is a bounded screen, not a claim that polarity alone causes a machine effect.

## Frozen scoring decision and result

After the visual exclusion and **before gallery/model access**, a separate protocol froze exactly three JPEG hashes: 032 chimera, 032 monotone control and 037 chimera. It required the calibrated development SFace, author GhostFaceNet and official SCRFD/ArcFace R50 pipelines, each with its frozen threshold; the seven export/JPEG/resize/crop/blur conditions; and the clean original plus three other same-person views as references. A person/model was eligible only if all four clean views were valid and the clean source matched in all seven conditions. Edited detection, selection or alignment failure would be inconclusive, never a nonmatch. The forward gate required **all** of: both chimeras valid on all seven conditions and three models; at least one chimera nonmatching all four own references in all seven conditions on all three models; and a `>=0.05` reduction in 032 ArcFace worst-condition own-gallery cosine versus its monotone control. There was no optimizer, rerender, new person, alternate strength or checkpoint selection.

All nine person/arm/model clean controls were eligible and matched in all seven conditions. All **63/63 edited evaluations were valid**, with zero inconclusives. Worst cosine is the maximum over all seven conditions and four same-person references; lower is better. `NM` counts conditions where **none** of those references matched.

| Identity / arm | SFace worst / NM (`0.515038` threshold) | Ghost worst / NM (`0.343449`) | ArcFace worst / NM (`0.237600`) |
| --- | ---: | ---: | ---: |
| 032 / eye-positive chimera | 0.662766 / 0 of 7 | **0.231306 / 7 of 7** | 0.494497 / 0 of 7 |
| 032 / monotone control | 0.713437 / 0 of 7 | 0.716080 / 0 of 7 | 0.850276 / 0 of 7 |
| 037 / eye-positive chimera | 0.734254 / 0 of 7 | 0.398249 / 0 of 7 | 0.504418 / 0 of 7 |

The 032 chimera reduced ArcFace worst cosine by **0.355779** versus the approved monotone control, exceeding that one margin criterion. ArcFace nevertheless matched both chimeras in all seven conditions, as did SFace. Only GhostFaceNet's 032 chimera gave seven nonmatches, for **7/63** nonmatching edited conditions overall. The required three-model nonmatch criterion failed. Retire this fixed palette/band setting; the result neither validates a protection claim nor tests the four reserved final recognizers (AdaFace, MagFace, EdgeFace, TransFace). ArcFace had already informed development research and is not an independent held-out model.

## Immutable local evidence and scorer provenance

Personal photos, processed JPEGs, scores, model weights and face crops remain outside Git under private `FCKFACE-data/runs/contrast-chimera-v1` (render) and `contrast-chimera-v1-development-screen` (scoring). The latter has the per-model original results, 63-row aggregate JSON and CSV. SHA-256 fingerprints:

| Artifact | SHA-256 |
| --- | --- |
| FRLL manifest | `2509c48bd7852251f7a3778cc7e1e320e17003d1bb56cca6dbf84998b4931600` |
| Clean 032 / 037 sources | `a9b07dec972d93f84449120fbd085f1ef4a4d4f76addac69751d9b596919df0b` / `1934782ca229c530b48efbe7fc44ee102a2d7965dfa2421b4e0eeb14a4ebaed1` |
| Executed renderer source / pre-render protocol | `1e3f09b3d4644dde2c97d26bcdc91738658d7235109b0b8f17ea9785d06a472c` / `061ca9cc2ec7de71b03c9d47cf934f2c6df6beeae119984382bf506669a1a952` |
| Six-output frozen appearance manifest | `4ead5e5e4a3abe3f4758539859b2a4089677779cf8fdb78f6829ed43dcd8cdad` |
| 032 chimera / monotone / full-negative JPEG | `4288e4363b9666acf9297dcb58ee71f4d6f52dd15eebfd34dda6302cba2e45f1` / `d846a5d1894af77035bbe1431cefd51de3c18455efe3250a998611453ff80548` / `1eeb6fdc0e56f355b93ed4efa1e86a628a435a3b450819591d33b3db6a7cb7f8` |
| 037 chimera / monotone / full-negative JPEG | `d839e534aebefd1b35ffd904d169cc971c1a30cff0b4b03a3f1b635e68344e18` / `a1eec4f1244c9cecf91cfa5ecec35ae5d138b75539a3c52147ae5262eea72586` / `c055bed90370848bc214b4d3d3d40c8a2056b5ad352d229c03573f738814d60f` |
| Three-row scoring manifest / pre-score protocol | `1c2d015ba93489f6a58f25cc4831c4e6383c78693d9280d0d8b641e129c1ce61` / `80151baec4e0edb1034c24da5b0683b2c2023fb3c6ffc148e94101f5442c42aa` |
| SFace / Ghost / ArcFace calibration JSON | `03f93365189dae853e306bdbed7a96536926628dfa1302d363d2779db98c8f04` / `7c0374b294bf20884bc3eba29b5d7a23b9629d778d382c90177ef4d633fbe90f` / `a186f3d285f4cff65d8c187a65f5929c19b13b86f8170583dee34dccab9bb56a` |
| SFace / Ghost / ArcFace result JSON | `f20748e849853375749d7adf1d63e7ba0dfaa73b437a0305d55549e264edb7c5` / `b734779e299da0b586c1ea0c245aa6242d9488232ca91eb4c42288347fb7b366` / `3dae8e2cc7701820fd1849b36c49b1fd928a71edd4872b7777435e330669732e` |
| 63-row aggregate JSON / CSV | `b4d54ad38cabf6253fa0e30c45cd0f9e1db3970031fe1665a858026194917423` / `6677e18d2fd25bc912fde1a125782f824dfef0f68ba109944708df0df6b3a8c3` |

The first SFace preflight stopped because its calibration pins the **entire** historical `models.json` SHA `eadf4841863d045a147d488239edc9c01d4d5c41f02f54df400d38bdda76ffc6` (Git `9297f55:research/models.json`), while the current registry SHA is `d67f5d0d44fa643bf52ea25ef891fa54c9582332b9805bb6d7f290bcad7a102c` after later final-panel metadata. The executed new SFace scoring wrapper then allowed the metadata difference by checking development entries and exact calibration-pinned source/model hashes; its **executed bytes** are preserved privately as `sface/executed-score_frozen_sface.py`, SHA `8e8057ca3cb0e760128ff10fbeafec8f71cec1fd51e8149da6769c7e2a9fc23c`. After scoring but **before inspecting score values**, a private provenance addendum (SHA `412939b1ab4d7c49d2b289caf560ccc494a4839d6b0c9d57d22cb35bb5a8b353`) recovered the exact historical registry, confirmed its SHA, compared the **entire development subtree** to the current registry, and rechecked six other calibration-pinned source hashes plus both model artifact hashes and sizes; all checks passed. SFace's runtime model class does not read `models.json`. This post-score verification must not be misrepresented as a pre-score full-file guard. GhostFaceNet's existing scorer retained its unchanged exact full-registry guard; the ArcFace scorer was unchanged.

For a **future new-output reproduction**, the current `research/score_frozen_sface.py` requires `--models-registry <private calibrated-models-9297f55.json>`, checks that file's exact calibration SHA and full development-subtree equality, and retains all runtime source/artifact checks. That revised wrapper passed preflight only; it did **not** generate the reported scores. Use a new output directory outside Git and the three-row frozen manifest, for example:

```powershell
$py = '<PYTHON_WITH_RESEARCH_DEPENDENCIES>'
$data = '<EXTERNAL_FCKFACE_DATA_ROOT>'
$weights = '<EXTERNAL_CALIBRATED_MODEL_DIRECTORY>'
$screen = "$data/runs/contrast-chimera-v1-development-screen"
& $py research/score_frozen_sface.py `
  --candidates "$screen/frozen-inputs.json" `
  --manifest "$data/frll/manifest.json" `
  --calibration "$data/runs/calibration-sface-v1/calibration.json" `
  --models-registry "$screen/sface/calibrated-models-9297f55.json" `
  --yunet "$weights/face_detection_yunet_2023mar.onnx" `
  --sface "$weights/face_recognition_sface_2021dec.onnx" `
  --output "$data/runs/contrast-chimera-sface-REPRO"
```

The archived scoring outputs and renderer source are retained unchanged. This reproduction command is documented, not an additional recognition run. No reserved final model was loaded.
