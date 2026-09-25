# H14 balanced loss and compact feature basis: development screen

**Both fixed mechanism comparisons failed their predeclared transfer gate.** The six images remain visible regular-dot effects with plausible facial structure in root screening, but every ArcFace condition still matched its own clean gallery. This is development evidence, not a privacy or release result.

## Fixed comparison

H14 held the H5 carrier constant: 14×14 radius-0.3 dots, 588 RGB coefficients, seed 0, face RMS target 16, channel cap 64, and Adam learning rate 0.15. Each arm used only one original `neutral_front` photograph from each of development identities 001 and 003. Four fixed clean-landmark objective conditions were export, JPEG75/4:2:0, blur and crop90. The loss margin for each model and condition was `(source cosine − frozen threshold)/(1 − frozen threshold)`; no other same-person view entered optimization.

| Arm | Two gradient models | Loss across four conditions |
| --- | --- | --- |
| A, `sg_global_max` | SFace + GhostFaceNet | One maximum over both models and conditions |
| B, `sg_balanced` | SFace + GhostFaceNet | Mean of the two per-model condition maxima |
| C, `s0095_balanced` | SFace + OpenVINO 0095 | Same balanced loss as B, substituting the compact feature basis |

Every arm made 18 forward/gradient passes, with **17 effective updates** in the unconditionally frozen step-18 JPEG: 144 edited model-condition gradient forwards per arm, 864 across six searches, and no native checkpoint queries. This matches the two-model forward budget while A/B tests loss aggregation and B/C tests the chosen second feature basis. The C arm is not a free extra model or more query budget. The 0095 Torch ONNX conversion was pinned to a native-forward/input-gradient diagnostic; the H14 prep independently matched the official integer-ROI, inverse-map nearest warp and linear BGR128 input pixels for both source images and all eight clean objective conditions. Its fixed-geometry float residual pullback is a BPDA approximation, not a true JPEG gradient. See [the 0095 calibration and adapter](openvino-development.md).

The six exports and their SHA-256 manifest froze before any other own-identity gallery view was read. Root then viewed full-photo and face-crop sheets and allowed all six to proceed: facial structure remained recognizable, the colored dots read as artificial, and no anatomy-like artifacts were seen. This was preliminary root screening, not independent human same-person validation. The seven-condition, four-development-model scoring protocol was recorded **before** scorer or gallery access. Each model used the clean original plus three separate own clean views; all references were valid, and every clean control matched in every condition. Detection or face-selection failure would have been inconclusive.

## Results and decision

All **168/168 edited model-condition evaluations were valid**. The 42 processed candidate JPEG hashes agreed exactly across all four scorers. The table gives the **maximum cosine over seven conditions and four own clean references** (lower is better), followed by the number of valid nonmatching conditions out of seven. Frozen thresholds: SFace 0.515038, GhostFaceNet 0.343449, ArcFace 0.237600, 0095 0.430598.

| Identity / arm | SFace cosine; NM | Ghost cosine; NM | ArcFace cosine; NM | 0095 cosine; NM |
| --- | ---: | ---: | ---: | ---: |
| 001 / A global | 0.378995; 7/7 | 0.176111; 7/7 | 0.466608; 0/7 | 0.318730; 7/7 |
| 001 / B balanced | 0.340187; 7/7 | 0.185707; 7/7 | 0.457518; 0/7 | 0.333550; 7/7 |
| 001 / C S/0095 | 0.355694; 7/7 | 0.337936; 7/7 | 0.508057; 0/7 | 0.063274; 7/7 |
| 003 / A global | 0.433299; 7/7 | 0.292154; 7/7 | 0.449066; 0/7 | 0.512765; 0/7 |
| 003 / B balanced | 0.313521; 7/7 | 0.457544; 0/7 | 0.523761; 0/7 | 0.546873; 0/7 |
| 003 / C S/0095 | 0.331400; 7/7 | 0.551096; 0/7 | 0.607925; 0/7 | 0.411799; 7/7 |

The predeclared mechanism gate required **at least 0.05 lower ArcFace worst-gallery cosine on each person**, with all seven conditions valid and each paired processed-condition face-RMS gap no greater than 0.25. B−A ArcFace reductions were **+0.009090 for 001 and −0.074695 for 003**; C−B reductions were **−0.050539 and −0.084163**. Both comparisons fail. C improved the targeted 0095 score on both people, but it worsened ArcFace on both and GhostFaceNet on 003. Neither B nor C shows the hypothesized broader transfer. No arm met the additional candidate-expansion condition of all-seven nonmatch on SFace, GhostFaceNet and 0095 for **both** people. Retire this fixed H14 setting without more steps, seeds or people.

All paired distortion checks passed: the largest B−A gap was 0.143592 for 001 and 0.169896 for 003; C−B maxima were 0.076982 and 0.010614. Exported face RMS was 16.052906–16.061895. The six searches took 19.39–24.04 seconds apiece, **137.13 seconds together excluding model loading**; this is native research timing, not a browser or phone benchmark. The carrier, two known people and recognition models were development-influenced. ArcFace is a development model, not an independent holdout. AdaFace, MagFace, EdgeFace and TransFace were untouched. No novelty, independent transfer, human-appearance, privacy or 95% release claim follows.

## Reproducibility and private evidence

The implementation is [`optimize_balanced_basis.py`](optimize_balanced_basis.py) with its [0095 prep and alignment helper](h14_0095_alignment.py). The prep runs separately in the accepted native OpenVINO environment before the source-only optimizer. Frozen native SFace, GhostFaceNet, ArcFace and [0095](score_frozen_openvino.py) scorers consume the same six-JPEG manifest afterward. The old SFace scoring wrapper takes the exact calibrated historical registry as an explicit input. Calibration thresholds, model artifacts and scorer sources are guarded; the source-only run snapshots its code and private model copies. Private photos, embeddings, model files, processed JPEGs and per-reference scores remain outside Git in `FCKFACE-data/runs/balanced-basis-v1-native-prep`, `balanced-basis-v1` and `balanced-basis-v1-development-screen`.

| Private artifact | SHA-256 |
| --- | --- |
| H14 optimizer / 0095 helper source | `246a54398ec4449c52b4a18a773f7e5549c0565db0e96ae01aee839f94728ebe` / `bb3e9ac953f2b2e880d6ed47dd70d35147c41be4fcec704e0fef4241ff17a3ca` |
| Native 0095 prep protocol / record | `012fe1265ef62b4c542c9d70fde32da70a97ecb45e5a3d30c2179b1cb2baeb33` / `648a96e0b00c5ca5995d2e2c35c36312248a0fc2dada26a8df4e156d6e570d03` |
| Optimization protocol / six-JPEG manifest | `bb78d04424f8b099dc080cfe4509c427266fdc73c99d41f8b642d5cc9abe3dab` / `512f8eb8976b3859b0eb236d36f10b17c092c3c76399e573b21edb3d9bf15773` |
| Pre-score appearance/gate protocol / aggregate disposition | `2b0de2536888b1c930c1f8e1bbd038c891837148d972781cbb404deb5152eb5b` / `6b582c28149e8bdd2bece1a13e3461bc1488d5c0ec1114a2bac5d56edebc27ea` |
| SFace / GhostFaceNet result JSON | `22374bf3ea6876ab839272da9cbbbed0b1a4a713e98fbf3d9804f8accd7a0fb5` / `c0426bd7292786418527a581f5c09d9c4ae49f660531ca6924d8116275fdc697` |
| ArcFace / OpenVINO 0095 result JSON | `a90dddd533f5a240e8d705d6af2d48d7d124ccb64ba67bc70c4053c1d9e752ae` / `8015a7480862ef4a4a92423d2484da6436bbb497425813af5f81ade881d86d2c` |

The six exact JPEG hashes are in the frozen private manifest; the aggregate disposition pins every scorer result and preflight. The 0095 ONNX gradient model SHA-256 is `8f9880452be0bc0842ed580123f79b93e145079b27d21cc867bd3208f4b695b3` (4,461,366 bytes). Native-gradient diagnostics passed on three synthetic tensors and two reviewed real crops before H14; H14's model-free fixed-ROI pullback directional finite-difference check passed with relative error 8.67×10⁻⁶. These component checks validate the implementation path, not transfer efficacy.
