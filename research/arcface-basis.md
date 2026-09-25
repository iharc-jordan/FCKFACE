# H15 SFace–ArcFace dot basis: development screen

**The fixed H15 setting failed its predeclared gate.** Both exported portraits passed preliminary root appearance screening as visibly artificial dots with plausible facial structure. Every edited ArcFace condition nevertheless matched its own clean gallery. This is development evidence, not independent human identity validation or a privacy result.

H15 tests a conventional white-box baseline: adding the already-development ArcFace gradient to the SFace objective on the same constrained dot artwork. It held the H14 carrier, identities 001/003, seed 0, 588 RGB coefficients, 14×14 radius-0.3 dots, face RMS 16, channel cap 64, Adam rate 0.15, and four source-only objective conditions (export, JPEG75/4:2:0, blur, crop90). The loss was the mean of the two models' maximum normalized source-cosine margins, `(cosine − calibrated threshold)/(1 − calibrated threshold)`. Each person used 18 forward/gradient passes, **17 effective updates**, 144 edited model-condition forwards, and no native checkpoint query. The step-18 JPEG was frozen unconditionally before H15 gallery scoring; optimization used only the source view. These identities and their other views had already informed earlier development experiments. Frozen H14 B (SFace/GhostFaceNet balanced) and C (SFace/0095 balanced) are compute- and distortion-matched controls; they were not rerun.

The ArcFace forward path used the official SCRFD detection and five-point alignment, then the pinned official ArcFace model. All ten source-only clean input crops/blobs were pixel-equal to the native path, and the converted graph's largest raw output difference was 7.51×10⁻⁶. The backward path held clean landmarks fixed and used a bilinear warp residual/BPDA approximation through edited JPEGs. This checks implementation consistency, not attack effectiveness. Model weights and embeddings remain private; the official ArcFace model's research terms do not grant permission to bundle it with the public tool.

Root reviewed the four full-photo/face-crop sheets before scoring: the dots read as artwork, geometry appeared plausible, and no anatomy-like artifacts were observed. That is screening, not an independent human same-person assessment. A separate frozen scoring protocol then used the exact two JPEGs, four calibrated **development** recognizers, each person's source plus three clean gallery views, and seven processing conditions. Invalid detection, alignment, or face selection would be inconclusive. The gate required all four models valid in all seven conditions, SFace **and** ArcFace nonmatching in every condition, at least one omitted model (GhostFaceNet or 0095) nonmatching in every condition **for each person**, and each processed face-RMS difference versus H14 B/C at most 0.25.

All **56/56 edited model-condition evaluations were valid** and both clean references were eligible. The 14 processed candidate JPEG hashes agreed across all four scorers. Each cell below gives worst cosine over seven conditions and four own clean references, then the number of **nonmatching** conditions out of seven. Thresholds were SFace 0.515038, GhostFaceNet 0.343449, ArcFace 0.237600, 0095 0.430598.

| Identity / arm | SFace | GhostFaceNet | ArcFace | 0095 |
| --- | ---: | ---: | ---: | ---: |
| 001 / H14 B | 0.340187; 7/7 | 0.185707; 7/7 | 0.457518; 0/7 | 0.333550; 7/7 |
| 001 / H14 C | 0.355694; 7/7 | 0.337936; 7/7 | 0.508057; 0/7 | 0.063274; 7/7 |
| 001 / H15 | 0.337245; 7/7 | 0.352860; 5/7 | 0.419155; 0/7 | 0.355450; 7/7 |
| 003 / H14 B | 0.313521; 7/7 | 0.457544; 0/7 | 0.523761; 0/7 | 0.546873; 0/7 |
| 003 / H14 C | 0.331400; 7/7 | 0.551096; 0/7 | 0.607925; 0/7 | 0.411799; 7/7 |
| 003 / H15 | 0.336770; 7/7 | 0.516732; 0/7 | 0.440566; 0/7 | 0.643576; 0/7 |

Including ArcFace in the source-only loss lowered its worst-gallery cosine relative to H14 B by 0.038363 and 0.083196 for 001/003, respectively, but **did not cross its calibrated match threshold in any condition**. On 001, GhostFaceNet gained two matches; on 003, both omitted models matched throughout. The maximum paired processed face-RMS gaps versus B/C were 0.03677/0.04022 for 001 and 0.05999/0.04937 for 003, well within the declared 0.25 balance limit. Retire this fixed H15 setting without more seeds, steps, or people. A score reduction is not a nonmatch, and this result cannot establish transfer to unused recognizers.

The two searches took 42.33 and 39.64 seconds excluding loading; the optimize phase took 92.27 seconds including model loading, with peak working set 1,445,457,920 bytes. These are native research timings, not browser or phone timings. ArcFace and all four scoring models have informed development; none is an independent held-out test. No novelty, release readiness, or 95% privacy claim follows.

## Reproduction and private evidence

The numerical sources are [`optimize_arcface_basis.py`](optimize_arcface_basis.py) and [`h15_arcface_alignment.py`](h15_arcface_alignment.py). With the already-pinned private FRLL manifest, calibrations, official model files and H14 control directory, run the optimizer's `preflight` phase, review `source-only-parity.json`, then pass its SHA-256 as `--expected-source-parity-sha256` to `optimize`. Both phases take `--manifest`, `--sface-calibration`, `--arc-calibration`, `--arc-dir`, `--arc-component-report`, `--h14`, `--yunet`, `--sface`, and `--out`. After reviewing the frozen JPEGs, pass `frozen-inputs.json` to the four [`score_frozen_sface.py`](score_frozen_sface.py), [`score_frozen_ghostface.py`](score_frozen_ghostface.py), [`score_frozen_arcface.py`](score_frozen_arcface.py), and [`score_frozen_openvino.py`](score_frozen_openvino.py) wrappers with their pinned calibrations and model assets. SFace needs the historical calibrated model registry; 0095 needs its frozen pipeline, erratum, executed calibrator snapshot, and official IR. The private run directories are `FCKFACE-data/runs/arcface-basis-v1` and `arcface-basis-v1-development-screen`; no photos or embeddings are in Git.

| Private or source artifact | SHA-256 |
| --- | --- |
| Optimizer / alignment helper | `d46b82c0fbcbd2f36303e503fe906e8276f7dd54ddc9336beb09631d70a2943d` / `99d3ecadf3b9e724b2cf7bd831c88094fff4b5f702cf62c8f54d92d9ccf7127f` |
| Source-only parity / optimization protocol | `3e6c8c1d5e10f5b9de8df15510107ea05419c413f15acf145dce01fe117559da` / `cec8bf7eb0ca4122d001852e020527acc85d17cc6cfcfa91be56e76eb827233b` |
| Frozen two-JPEG manifest / freeze result | `1e356550b02715cb800d6322b6703a637a5ce7bd8911d5561ab5327599a3f01d` / `3e82df24758465805b0cb38bc662d79e9b4e05149992b12b5c370047f933c49f` |
| Scoring protocol / checked aggregate | `e05299268c75e95850e34c9920ecf88707e337776e34b27c3b2816124b54cc8a` / `580771938a936e693fcf5879ebcc799c530b638c2dca48be8b437e79023c8a92` |
| SFace / GhostFaceNet results | `d258e21d86766b85d0c3c78f2f32dd183d9e82b835a716b7d3b342c7e2f7d912` / `9da22ad7a4b19f83f7de1cdf28ca016277008bf69fcc17fd3957e850555b8595` |
| ArcFace / 0095 results | `145e5cdd657e0893cc70bc4963fd2ef795dc1c05a8e4af1effc1c5a41028f741` / `452a0134afdd0dd4449663f3bd296e432c03ffd6e00802b8d6257e05bec14c4c` |

The exact exported SHA-256 values are `a18684846574a0f53618fc14399b1f3b212985f6291e2ee27905649787f203d1` (001) and `e7069bb769222e6d2607f6426e05a6fa3d73aa606ee2e5f1df49b46351c5245e` (003). The private result files retain per-condition, per-reference scores and exact processed JPEGs.
