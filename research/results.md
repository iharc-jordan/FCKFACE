# Development findings — 25 September 2026

**No method meets the release requirements.** These are small development experiments against SFace and GhostFaceNet, both used in earlier work. AdaFace, MagFace, EdgeFace and TransFace have not been evaluated. Independent human appearance review and real-phone testing are pending.

The [protocol](README.md) defines the frozen model, preprocessing and seven processing conditions. Each comparison uses the exact exported JPEG, fresh detection/alignment, and the original plus three separate clean references. A nonmatch requires every reference to fall below the calibrated threshold; a failed detection cannot count as a nonmatch. Repeated seeds, parameters and processing conditions are correlated observations, not additional people.

## Calibration

FRLL's separate calibration split contains 20 identities and 200 images. Native preprocessing produced 173 valid embeddings; 27 missing detections remained inconclusive. Of 19,900 possible pairs, 14,197 impostor and 681 genuine pairs were valid, for 74.76% pair coverage. At a target false-match rate of 0.001, the frozen cosine threshold is **0.5150383510**. Empirical false-match and false-nonmatch rates are 0.0009861 and 0.0969163. Leaving one identity out changes the estimated threshold from 0.488984 to 0.522429. This limited calibration is not a population guarantee.

The second development recognizer, GhostFaceNet, loads its serialized author H5 graph directly with its saved mixed precision, author RGB/skimage alignment and normalization, and a declared YuNet detector substitution. On the same calibration identities it produced 173 valid images, the same valid-pair coverage, and a separate threshold of **0.3434492487**. Empirical valid-pair FMR was 0.0009861 and FNMR 0.01762; leave-one-identity-out thresholds ranged 0.337702–0.344438. See [the model and preprocessing record](development-models.md). Neither calibration establishes population performance or held-out effectiveness.

## Fixed effects and selective reconstruction

| Experiment | People / candidates | Valid processed-image comparisons | Gallery nonmatches | Decision |
| --- | ---: | ---: | ---: | --- |
| Initial three-family pilot | 2 / 88 | 616 / 616 | 0 | Do not expand unchanged regional/detail effects |
| Multiscale artwork, including pilot | 8 / 96 | 672 / 672 | 0 | Retire the fixed pattern settings |
| Opaque graphic detail bottleneck | 2 / 12 | 84 / 84 | 0 | Retire these six-panel settings |

The artwork count includes pilot cases and must not be added to the pilot as independent evidence. Landmark-relative artwork had lower maximum gallery cosine than the image-anchored control in 199 of 224 paired conditions, with a mean difference of −0.02835. However, effective pattern frequency differed: fitted-to-control coordinate scale ratios ranged from 0.754 to 0.843. That confound prevents attributing the change to anchoring. Four uniform-dot renderings also missed their requested RMS budget.

The detail bottleneck removed fine detail before four-colour, opaque, eight-pixel tile reconstruction in six facial panels. It was compared with reconstruction from unblurred input and fixed tiles. Both people had valid clean controls. Treatment-minus-control average cosine differences were nonnegative in all four sigma/control comparisons: no consistent benefit from intentional detail removal. Post-JPEG face RMS ranged about 34.3–37.8, with paired differences up to 1.23; these were coverage-matched, not exactly distortion-matched. Root review found the blue/purple panels clearly artificial; this is not independent human identity validation.

## Regional optimization with extra references

This diagnostic optimized against four clean photos of each person, including reference photos unavailable to a one-photo user. It is an **oracle experiment**, not a deployable method. Each joint or whole-face search proposed at most 64 candidates. Independent-region and best-single controls shared an eight-candidate sign search because RMS projection makes single-region magnitude redundant; they are not separate replications or equal query-budget controls.

Worst cosine across all seven conditions and four references follows. Lower is better; values below 0.5150383510 are nonmatches.

| Development identity / face RMS | Fixed artwork | Joint regions | Independent combined | Best single region | Whole-face 2×2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 001 / 8 | 0.89979 | 0.84848 | 0.90879 | 0.88528 | 0.81777 |
| 001 / 16 | 0.82062 | 0.73468 | 0.82062 | 0.76874 | 0.72384 |
| 003 / 8 | 0.89104 | 0.74748 | 0.85941 | 0.83233 | 0.82246 |
| 003 / 16 | 0.66768 | **0.48774** | 0.65624 | 0.67195 | 0.58418 |

All 140 final processed-image comparisons were valid. There was one successful identity/RMS/arm case. The run made 542 distinct objective evaluations from 544 proposals in 439.6 seconds. This small result justifies a bounded single-photo ablation; it does not establish transfer to new people, other models, mobile runtime, or acceptable appearance. Root review inspected complete photos and crops. No independent human reviewers have evaluated these images.

The subsequent source-only ablation at RMS 16 selected exactly the same JPEG bytes and coefficients for both joint and whole-face arms on both people. It froze all selections before opening extra same-person reference images. A confirmation on the remaining six prespecified people added **zero successes**. Combined results: joint regions **1/8 (12.5%)**; whole-face 2×2 **0/8**. All eight clean controls were eligible, with 56/56 valid edited conditions per arm and no edited inconclusives. Retire these unchanged settings.

Both arms together required a mean 103.46 seconds of native CPU search per photo (92.43–113.64 seconds). This is the cost of running two experimental arms, not a measured browser runtime or the cost of one selected method. Pilot plus confirmation made 1,024 proposals and 1,021 distinct objective evaluations in 865.82 seconds including final evaluation. These are development-influenced SFace results, not independent release validation.

## Identity-stable subspace loss

This hypothesis asks whether suppressing embedding directions that are stable across photographs helps more than suppressing the whole embedding. A between/within-identity covariance fit used 52 other development identities and four views each, excluding all eight screen identities, calibration and final identities. Rank-8 and rank-16 projections were compared with random projections matched in rank and singular spectrum. Training diagnostics do not establish transfer.

For each of the two pilot people, four source-only searches used the same regional renderer, RMS 16, 64 proposals, and a loss equally weighting full-embedding cosine and projected cosine. Every selected JPEG was frozen before opening the separate gallery. Worst gallery cosine across all seven conditions follows:

| Development identity | Learned rank 8 | Random rank 8 | Learned rank 16 | Random rank 16 | Earlier full-embedding loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 001 | 0.75461 | 0.76829 | 0.72052 | 0.70906 | 0.73468 |
| 003 | **0.50509** | 0.53163 | 0.57618 | **0.46580** | **0.48774** |

All 56 final conditions were valid. Learned rank 8 added no successful person over the original loss; random rank 16 was better on the same successful person. There is no demonstrated benefit from the learned identity-stable directions. Retire this setting rather than expanding its grid. Face RMS was 16.000–16.003; 512 proposals were used across the eight searches.

## Whole-face poster screening

A source-only preview replaced photographic facial shading with four fixed blue/purple levels, compared with unfiltered posterization, continuously coloured shading and grayscale smoothing. Four preselected development identities, two smoothing presets and four arms produced 32 JPEGs. Root visual review rejected the poster treatment before recognition scoring: too much individual facial detail was lost to support unequivocal identity. Continuous-colour controls retained more likeness, but do not establish the proposed detail-removal mechanism. No recognition success rate is available for this rejected preview. Face RMS differed substantially between arms, so it is not a distortion-matched comparison.

A changed contour treatment preserved source-derived feature outlines and dark eye/nose/mouth regions on cyan paper. The same four identities, two stroke widths and three backgrounds produced 24 further previews. Feature outlines became more legible, but root review still found individual identity uncertain, so this setting was also rejected without recognition scoring. Neither root screen substitutes for independent blinded human review. Photograph-to-drawing translation has established [geometry and semantic preservation prior art](https://arxiv.org/abs/2203.12691); artwork by itself is not a novelty or protection claim.

## Gradient implementation check

The official SFace ONNX converted through `onnx2torch` matched native OpenCV on one deterministic synthetic input: maximum raw-feature difference 2.38×10⁻⁶ and normalized-feature cosine 0.9999999999968. A finite-difference directional derivative agreed with the analytic gradient within 1.57×10⁻⁷ at an input L2 step of 1. Two separate processes reproduced the numerical records. CPU forward/backward took approximately 0.040/0.060 seconds; working set at the end was about 448 MB. This verifies one gradient reference, not photo-pipeline parity, protection, or browser performance.

## Independently optimized dots

The next source-only pilot compared unrestricted RGB pixels, a smooth 14×14 RGB field and independently coloured circles on a regular 14×14 grid. Each arm used 48 gradient steps and six native selection checkpoints, with face RMS 16 and a maximum channel change of 64. Forward optimization used actual JPEG/blur pixels at fixed clean alignment; an approximate backward pass ignored the derivatives of JPEG, rounding and blur. Selection used fresh native detection and alignment on the exact exported images. All six choices were frozen before opening separate gallery photos.

Both clean photos were eligible. All 42 candidate conditions were valid for each recognizer. Worst clean-gallery cosine across the seven conditions follows; the two models have different thresholds and must be judged separately.

| Identity / arm | SFace worst cosine (threshold 0.51504) | GhostFaceNet worst cosine (threshold 0.34345) |
| --- | ---: | ---: |
| 001 / unrestricted RGB | 0.81030 | 0.7900 |
| 001 / smooth field | **0.29511** | 0.4570 |
| 001 / regular dots | **0.29995** | 0.4908 |
| 003 / unrestricted RGB | 0.76029 | 0.8033 |
| 003 / smooth field | **0.12695** | 0.5121 |
| 003 / regular dots | **0.24730** | 0.6202 |

Dots and smooth fields each passed all SFace conditions on both people. **Every GhostFaceNet condition still matched**, including the source and smiling reference each time. The SFace-only effect therefore did not transfer to this second development recognizer. Dots did not outperform the smooth control on the binary SFace outcome. Root full-photo/crop review found the regular circles clearly artificial; smooth colour patches looked bruise-like and are unsuitable as a product candidate. This is screening, not independent human acceptance.

Post-export face RMS was 16.06–16.07 for dots, 16.02–16.03 for smooth fields and 16.49–16.54 for unrestricted pixels. Distortion changed differently under later processing: dots ranged 15.15–16.15, smooth fields 16.01–16.07, and unrestricted pixels 7.40–17.09. The arms have matched pre-export budgets and step counts, not identical distortion in every processed condition. Native dot searches took 38.07 and 36.24 seconds; browser optimization is unmeasured. An initial coordinate bug stopped before any gallery scoring, after two valid controls. Those controls and their exact executed code were preserved during correction; their individual runtimes and the failed-run total were not recorded. The resumed run took 170.88 seconds for the remaining work.

## Joint-model dot optimization

H6 kept the same dot geometry, RMS 16, channel cap and seed, and changed the loss to the worst model-specific margin `(cosine - threshold) / (1 - threshold)`. GhostFaceNet-only optimization used 48 steps; joint SFace/GhostFaceNet used 24. Both used 144 gradient model-condition evaluations and six native checkpoints, but the joint arm required 36 native model-condition queries versus 18 for Ghost-only. This matches gradient-evaluation count, not total compute. The frozen H5 SFace-only dots provide a third comparison without rerunning them.

The serialized GhostFaceNet graph uses mixed precision. A separate float32 clone retained all weights and 82 BatchNorm epsilon values, with normalized-feature cosine at least 0.999968 on the synthetic and two clean development checks. A finite-difference directional derivative agreed within 3.58% at an input L2 step of 0.5. This clone supplied gradients only; the original author H5 selected checkpoints and produced final scores. A post-run audit of 12 selected objective crops found clone/author cosine 0.999968–0.999985. These are local diagnostics, not browser parity.

All four new JPEGs were frozen before the separate clean gallery was read. All clean controls matched, and all 56 edited model-condition evaluations were valid. Worst gallery cosines follow:

| Identity / optimization model | SFace (threshold 0.51504) | GhostFaceNet (threshold 0.34345) | Both models nonmatch in every condition |
| --- | ---: | ---: | --- |
| 001 / GhostFaceNet | **0.50247** | **0.17758** | Yes |
| 001 / joint | **0.44340** | **0.22266** | Yes |
| 003 / GhostFaceNet | 0.58389 | **0.28225** | No |
| 003 / joint | 0.51558 | **0.31982** | No |

The joint result for 003 failed only SFace's crop condition: its smiling clean reference scored 0.5155815, above the frozen 0.5150384 threshold. The original-source score alone was below threshold, illustrating why separate reference photographs matter. The joint arm therefore did **not** meet the predeclared two-person, two-model, all-condition pilot criterion, and did not add a successful person over Ghost-only. Its partial improvement motivates a separate alignment experiment, not an expanded claim for H6.

Root full-photo and crop review found artificial regular circles with plausible likeness; independent human appearance review is still pending. Post-export RMS was 16.04–16.06. Ghost-only search took 134.91/128.30 seconds and joint search 74.68/79.52 seconds on native CPU, with 453.93 seconds for the complete experiment. Neither meets a demonstrated one-minute browser workflow. Checkpoint step numbers refer to the forward pass before that step's Adam update.

## Released line-drawing model

As a prior-art comparison, the authors' [Informative Drawings](https://github.com/carolineec/informative-drawings) generator processed a 512-pixel crop with facial context, then composited its grayscale output inside the selected face. Two released styles on the four preselected people produced eight fixed JPEGs. Root review found more individual structure than the handmade stencils, especially in the lighter style, and allowed a bounded recognition check. This is not independent human acceptance or a new FCKFACE invention.

All four clean controls were eligible. **None of the eight edits passed all seven SFace conditions.** Six outputs had valid recognition in every condition but still matched in at least one. Both styles for the fourth person failed face detection in every condition and remain inconclusive. Of 56 edited conditions, 42 were valid and 14 inconclusive. These unchanged settings are retired. CPU inference took 1.74–2.29 seconds per crop, with about 680 MiB peak process working set; neither runtime nor appearance establishes protection.

## Reproducibility

Runners, selection rules and model hashes are in this repository. Local run directories retain source snapshots, exact images, all reference scores, failures and timings outside Git. SHA-256 fingerprints of the completed detailed reports are:

| Report | SHA-256 |
| --- | --- |
| Combined artwork analysis | `bf21492168ab9ed5a64d6340d9669daa5a2a90c1aa1a6d25ab88ee2655657848` |
| Regional oracle results | `1b2bda72eca3b9fe1d784133345ec1dddd009a4b34201045790ecb9e72a406b4` |
| Graphic bottleneck scores | `474380712fcdd35b386b0c4b44457b1d0d8681695d8d78b2c7ed58835451b03d` |
| Source-only regional eight-person summary | `1b9b4741ea16e5bab79ba61e7f621ff185fa8c2399cf71608bd984c03961bb28` |
| Subspace-loss results | `d49d933fc405c5e68750b0e6b3eb365aa40f0d0071b957383682df1f1c52c018` |
| GhostFaceNet calibration | `7c0374b294bf20884bc3eba29b5d7a23b9629d778d382c90177ef4d633fbe90f` |
| Gradient artwork / native SFace | `0597e91bdff62c88cf26436210f5dafec4c795cb7eafb1b381ea2355f0ded1f3` |
| Frozen gradient artwork / GhostFaceNet | `00bb2acd6332a53d38ad5909be141e44a447b2cdcf2f66045da6354c70568b52` |
| Joint-model dots / both native recognizers | `40cd21968ec24a69739747e50a053cb02aca3c0b8eac9dd7c10176990ea561d7` |
| Released line-drawing model / SFace | `f105499fefc90b65a318d02c0cefc0ac11e5c6e8c5dcd8adb14c1ed8080a589f` |

The detailed reports contain local biometric artifacts and are not distributed. The aggregate findings above are the public record; no release success rate is claimed.
