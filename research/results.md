# Development findings — 25 September 2026

**No method meets the release requirements.** These are small development experiments against SFace, GhostFaceNet, ArcFace and OpenVINO 0095. AdaFace, MagFace, EdgeFace and TransFace have not been evaluated. Independent human appearance review and real-phone testing are pending.

The [protocol](README.md) defines the frozen model, preprocessing and seven processing conditions. Each comparison uses the exact exported JPEG, fresh detection/alignment, and the original plus three separate clean references. A nonmatch requires every reference to fall below the calibrated threshold; a failed detection cannot count as a nonmatch. Repeated seeds, parameters and processing conditions are correlated observations, not additional people.

## Calibration

FRLL's separate calibration split contains 20 identities and 200 images. Native SFace preprocessing produced 173 valid embeddings; 27 missing detections remained inconclusive. Of 19,900 possible pairs, 14,197 impostor and 681 genuine pairs were valid, for 74.76% pair coverage. At a target false-match rate of 0.001, the frozen cosine threshold is **0.5150383510**. Empirical false-match and false-nonmatch rates are 0.0009861 and 0.0969163. Leaving one identity out changes the estimated threshold from 0.488984 to 0.522429. This limited calibration is not a population guarantee.

The second development recognizer, GhostFaceNet, loads its serialized author H5 graph directly with its saved mixed precision, author RGB/skimage alignment and normalization, and a declared YuNet detector substitution. On the same calibration identities it produced 173 valid images, the same valid-pair coverage, and a separate threshold of **0.3434492487**. Empirical valid-pair FMR was 0.0009861 and FNMR 0.01762; leave-one-identity-out thresholds ranged 0.337702–0.344438. See [the model and preprocessing record](development-models.md). Neither calibration establishes population performance or held-out effectiveness.

The additional [OpenVINO 0095 development calibration](openvino-development.md) uses the same 20 calibration identities and a separately pinned Intel demo pipeline with YuNet substitution. It yielded 173 valid images, 681 genuine and 14,197 impostor pairs, and 5,022 invalid pairs. Its frozen threshold is **0.4305980817**, with 14 false matches and 52 false nonmatches; leave-one-identity-out thresholds ranged 0.407524–0.433166. The record explains a corrected metadata description: the executed alignment uses nearest-neighbor warp followed by linear resize. No reserved model was used.

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

## Condition-specific alignment

H7 tested whether refreshing landmarks on edited images helps more than fixed clean landmarks for each processing condition. Both arms used the same regular dots, RMS 16, channel cap 64, seed, 18 gradient steps and four objective conditions (export, JPEG75, blur and crop90). Each arm made 144 gradient model-condition evaluations, six checkpoints and 48 native selection queries. The refresh arm updated edited-image landmarks every three steps; detector coordinates were treated as constants in the backward pass. Every selected JPEG was frozen before separate gallery evaluation.

Both clean controls were eligible; all 56 edited model-condition evaluations were valid. Worst gallery cosine across the seven conditions follows:

| Identity / alignment | SFace (threshold 0.51504) | GhostFaceNet (threshold 0.34345) | Native search seconds |
| --- | ---: | ---: | ---: |
| 001 / fixed per condition | **0.40482** | **0.19269** | 44.17 |
| 001 / refreshed | **0.40143** | **0.20201** | 52.19 |
| 003 / fixed per condition | **0.43742** | **0.28855** | 43.93 |
| 003 / refreshed | **0.41477** | **0.23874** | 46.12 |

Both arms passed both development models in every condition on these two people. Refresh added no binary success and did not uniformly lower scores. These results do not demonstrate a general benefit from refresh. Comparison with H6 does not isolate the effect of cropping: the objective conditions, step count, alignment treatment and native query count changed together.

The native search used a compiled TensorFlow gradient function after a synthetic eager/compiled check (value difference zero; maximum gradient difference 2.44×10⁻¹¹). Search timings exclude model loading and initial compilation and are not a one-minute browser workflow. Total experiment wall time was 229.56 seconds; post-export face RMS was 16.03–16.07. Root reviewed full photos and crops and found plausible likeness with artificial dots. Independent human acceptance and transfer to other recognizers remain unestablished.

An independently calibrated third development pipeline, official InsightFace SCRFD plus ArcFace R50, then evaluated the same four frozen JPEGs. All 200 calibration images were valid; the threshold was 0.2376004863 at an empirical false-match rate of 0.001. **All four edits still matched in every one of the seven conditions**, with no inconclusives. Worst gallery scores were 0.463426/0.491219 for fixed/refreshed 001 and 0.446488/0.406986 for 003. Both clean controls were eligible. The current two-model dot setting therefore fails this transfer screen and will not be expanded unchanged. See the [pinned ArcFace pipeline and results](arcface-development.md). None of the four reserved final recognizers was used.

The later [H9 alignment diagnostic](alignment-diagnostic.md) held ArcFace's network and official clean gallery fixed while swapping only the four H7 query crops from SCRFD to YuNet landmarks. Median maximum-gallery cosine fell by 0.015, below the predeclared 0.05 criterion, and two of four edits worsened. All still matched. The canonical scores reproduced the earlier export scores within 2.9×10⁻⁸. These counterfactual queries do not replace official model results; the diagnostic did not support the proposed alignment-jitter experiment.

A subsequent [OpenVINO 0095 diagnostic](openvino-development.md) scored these same four H7 JPEGs without changing them. Both clean controls were eligible and all 28 edited conditions were valid; their processed JPEG hashes matched the original H7 audit. For 001, both fixed and refreshed edits failed matching to every clean reference in all seven conditions, with worst cosines 0.305917 and 0.308577 (threshold 0.430598). For 003, both edits matched in all seven, with worst cosines 0.495844 and 0.581495. This establishes limited transfer to an additional development model for one of two people, not a reliable method or independent release evidence. The diagnostic took 9.578 seconds with 272 MB peak process working set; it was not a generation or browser benchmark.

## Cross-model identity-relation loss

H8B represented each model's identity embedding by its similarities to 52 other development identities, each represented by four clean views. It tested whether making SFace and GhostFaceNet move together in this common relation space improved transfer. Controls were the same source-match loss without the relation term and the relation term with shuffled identity correspondence. All arms used the same dots, RMS 16, channel cap 64, seed, 18 steps and four objective conditions. Every arm unconditionally froze its step-18 forward JPEG before separate-gallery scoring; six native checkpoints were diagnostic only. The [method and reproducibility note](relational-dots.md) records the formula, synthetic gradient checks, failed original execution and isolated rerun.

| Identity | Baseline ArcFace worst gallery cosine | Relational loss | Shuffled correspondence |
| --- | ---: | ---: | ---: |
| 021 | 0.592979 | 0.609110 | 0.603247 |
| 026 | 0.628492 | 0.636106 | 0.630162 |

Both clean identities were eligible on all three development models. ArcFace detected and matched all six exports in every condition: **0/42 nonmatching conditions**. The relational arm was worse than both controls for each person, failing its predeclared median improvement of at least 0.02 and both-person improvement. It is retired without expansion or reseeding. SFace/Ghost found no face in any of 021's baseline or relational conditions; the shuffled control was valid in four of seven. Those failures are inconclusive. All 026 conditions were valid, but every arm still matched in at least one condition on each model.

Exported face RMS was 16.036–16.073, with within-person arm spread at most 0.030. The six native optimizations took 328.4 seconds after anchor construction, 49.2–59.6 seconds per arm. Root reviewed all full photos and face crops and found plausible likeness with artificial dots; independent human appearance acceptance remains untested. The four reserved final models remain untouched.

## Momentum and input-diversity dot optimization

H10 compared Adam, momentum sign descent, and momentum plus random resize/pad of the **edited aligned crop**. All arms used the same untransformed native clean source embedding, fixed H7 dots (RMS 16, cap 64), seed 0, 18 steps, four gradient conditions, 144 edited model-condition forwards and 48 native diagnostic queries per arm. The forward step-18 JPEG was frozen unconditionally for each of two development people before opening their separate gallery views. The [protocol and reproducibility note](transfer-dots.md) records the exact schedule, implementation, hashes and prior art. Momentum and input diversity are established transfer-attack components, not FCKFACE inventions.

| Identity | Adam ArcFace worst gallery cosine | Momentum | Momentum + input diversity |
| --- | ---: | ---: | ---: |
| 029 | **0.448252** | 0.502479 | 0.507174 |
| 030 | **0.588494** | 0.605418 | 0.612109 |

Both clean people were eligible on SFace, GhostFaceNet and ArcFace. All 126 edited seven-condition model evaluations were valid; **ArcFace still matched all 42 candidate conditions** against four own-gallery views (threshold 0.2376004863). Against Adam and momentum respectively, the median ArcFace worst-gallery cosine *reduction* from adding input diversity was **−0.041268** and **−0.005693**, with negative reductions on both people. This failed the predeclared gain of at least +0.02 versus each control and positive gain on both people; retire the unchanged setting without expansion or reseeding. All three arms passed every condition on both native models for 029, but none passed all conditions on either native model for 030. There were no final detector or selection failures to misclassify as nonmatches.

Export face RMS was 16.050–16.075; subsequent processing ranged 15.168–16.143, with at most 0.078 within-person, same-condition RMS spread across arms. The six searches took 240.1 seconds excluding model loading, about 38.0–40.4 seconds per arm. Root review found visible artificial dots with facial structure plausibly intact; independent human identity acceptance remains pending. No reserved final recognizer was used.

## Sparse versus inverse dot support

H11 compared literal radius-0.3 dots with their exact complement **inside the same 14×14 grid**, using the same 588 RGB coefficients, seed, Adam update, face RMS 16 and channel cap 64. Each arm made three forward/gradient steps (24 edited model-condition forwards) and one native diagnostic checkpoint; its frozen step-3 forward JPEG reflected **two prior updates**, because the third update occurred after that JPEG was made. All four JPEGs froze before other same-person photos were read. The [protocol and reproducibility note](support-probe.md) records support counts, source hashes and the fixed gate.

| Identity | Dots ArcFace worst gallery cosine | Inverse dots | Inverse reduction | SFace/Ghost normalized-margin worsening |
| --- | ---: | ---: | ---: | --- |
| 001 | **0.556686** | 0.617093 | −0.060407 | +0.151425 / +0.204634 |
| 003 | 0.603987 | **0.596368** | +0.007619 | +0.138188 / −0.075040 |

Both clean people were eligible on all three **development** models. All 84 edited model-condition evaluations were valid, but all 28 ArcFace conditions matched own identity. Inverse dots missed the predeclared improvement of at least +0.05 on **each** person; three of four native-margin comparisons also worsened beyond +0.02. Retire only this bounded three-step setting, not every possible broader-support method. The exact exported face RMS was 16.048–16.058, and the largest processed-condition arm difference was 0.140 RMS, below the 0.25 imbalance flag. The inverse arm changed 86.2–86.8% of face pixels after export versus 51.6–52.0% for dots: matching RMS does not separate spatial support from local amplitude distribution. Root found the pattern clearly artificial with plausible face structure, which is screening rather than independent human acceptance. ArcFace was excluded from gradient optimization but had already informed development research; it is not an independent holdout. No reserved final recognizer was used.

## Eye-positive contrast chimera

H12 tested a continuous-tone blue graphic treatment on two development identities. The clean source eyes/eyebrows stayed positive while the rest of the oval face had reversed luminance order. Its appearance rationale came from [human contrast-chimera research](https://pmc.ncbi.nlm.nih.gov/articles/PMC2664053/), which did not establish machine-recognizer efficacy. The same-support monotone control was matched within predeclared tolerances on exported-JPEG face RMS and changed-face fraction. This was a different, much stronger distortion regime than RMS-16 dots: face RMS was 57.270 for 032 and 61.962 for 037. The [protocol, formulas and reproducibility note](contrast-chimera.md) record the six fixed JPEGs and private hashes.

Root reviewed both full photos and face crops **before any H12 recognition scoring**. The two chimeras and 032 monotone control had plausibly intact facial structure with an overt artificial effect. Both full negatives had uncertain eyes/likeness. The 037 monotone control erased substantial detail and was rejected, despite near-identical numerical RMS/coverage to its chimera; its monotone mapping clipped 55.37% of treated pixels. The three rejected JPEGs were preserved but never scored. This is root screening, not independent human identity confirmation; no isolated polarity inference is available for 037.

The subsequent development-only protocol froze exactly the three approved JPEG hashes, seven processing conditions, four own clean references, calibrated SFace/GhostFaceNet/official ArcFace, clean eligibility, and the forward gate **before gallery/model access**. All nine clean controls were eligible and matched in all seven conditions. All **63/63 edited model-condition evaluations were valid**, with no inconclusives. Nonmatching conditions: SFace **0/21**, GhostFaceNet **7/21** (all seven for the 032 chimera only), ArcFace **0/21**. Worst own-gallery cosine follows; thresholds were 0.515038, 0.343449 and 0.237600 respectively.

| Identity / arm | SFace | GhostFaceNet | ArcFace |
| --- | ---: | ---: | ---: |
| 032 / eye-positive chimera | 0.662766 | **0.231306** | 0.494497 |
| 032 / monotone control | 0.713437 | 0.716080 | 0.850276 |
| 037 / eye-positive chimera | 0.734254 | 0.398249 | 0.504418 |

The 032 chimera improved ArcFace worst cosine by **0.355779** over its matched approved control, exceeding the gate's 0.05 margin component, but ArcFace and SFace still matched both chimeras under every condition. Neither chimera met the required all-seven nonmatch on all three models. **Retire this fixed setting without expansion.** ArcFace was already development-influenced; no reserved final recognizer was touched and no privacy or release claim follows. An SFace calibration-registry metadata mismatch was resolved with an exact historical-registry/full-development-subtree provenance addendum after scoring, before score values were inspected; the executed wrapper is preserved. The current SFace wrapper requires an explicit exact calibrated registry; it passed H12 preflight after scoring but did not generate H12's results. See the [full provenance account](contrast-chimera.md).

## Chimera substrate versus stacked dots

H13 tested whether optimizing dots while the H12 graphic chimera is present transfers better than optimizing the same feasible dots on the original photograph and then stacking them onto the chimera. Both arms used the literal H5 14×14 radius-0.3 dots intersected with H12's hard treatment support, the same 588 seed-0 RGB coefficients, channel cap 64, and per-channel headroom valid on **both** substrates. The original eye band and outside-support pixels stayed unchanged before JPEG. A shared bounded pre-JPEG face-RMS projector ran during search; a single score-blind exact-JPEG scalar correction of the frozen step-18 coefficients matched the exported target. Both final images were chimera plus the learned field. Each arm made 18 forward/gradient steps, **17 effective Adam updates**, 144 edited model-condition gradient forwards and **zero native checkpoint queries**. All four JPEGs froze before other same-person views were read. The [H13 protocol and evidence note](chimera-dots.md) records formulas, controls, hashes, runtimes and rerun commands.

Root's preliminary full-photo/crop screen found all four edits overtly artificial with plausible same-person structure; that is not independent human acceptance. All 12 arm/model clean controls were eligible. Across 84 edited model-condition evaluations, **40 were valid and 44 were inconclusive no-face detections**, never counted as protection. All **28 ArcFace queries were valid and matched**. Worst valid-condition own-gallery cosine is shown below; `—` means no valid recognition result. SFace and GhostFaceNet each had only six valid conditions, all for 037 joint; its seventh crop90 was inconclusive.

| Identity / arm | SFace (valid/NM) | GhostFaceNet (valid/NM) | ArcFace (valid/NM) |
| --- | ---: | ---: | ---: |
| 032 / stack | — (0/0) | — (0/0) | 0.552359 (7/0) |
| 032 / joint | — (0/0) | — (0/0) | 0.490889 (7/0) |
| 037 / stack | — (0/0) | — (0/0) | 0.526907 (7/0) |
| 037 / joint | 0.640661 (6/0) | 0.354227 (6/2) | 0.583239 (7/0) |

For 032, joint lowered ArcFace worst cosine by **0.061469** versus stack, but only **0.003608** versus the unchanged H12 base, below the predeclared 0.05 reduction against both controls. For 037, joint worsened versus stack by 0.056332 and H12 by 0.078821. The all-seven-valid and joint native-model nonmatch gates also failed. **Retire this fixed H13 setting without expansion.** The maximum paired same-condition processed face-RMS difference was 0.02922/0.04834 for 032/037, below the 0.25 imbalance flag. Each search took 31.08–33.85 seconds excluding model loading; this is not a browser benchmark. ArcFace was already development-influenced, and no reserved final recognizer or independent human identity test was used. No novelty, privacy or release claim follows.

## Balanced loss and compact feature basis

H14 held the regular RMS-16 dot carrier and two-gradient-model budget fixed while comparing A, SFace/GhostFaceNet global-max loss; B, their mean per-model condition-max loss; and C, the same balanced loss with compact OpenVINO 0095 replacing GhostFaceNet. Each of the six searches on development identities 001 and 003 used 18 forward passes, **17 effective Adam updates**, 144 edited model-condition gradient forwards, and no native checkpoint queries. Six step-18 JPEGs froze before gallery access. Root's preliminary full/crop review accepted all six as visibly artificial with facial structure intact; this is not independent human validation. The [full protocol and evidence note](balanced-basis.md) gives the alignment/parity safeguards, controls, artifact hashes and limitations.

All four calibrated development models had valid clean references and matched clean controls. **All 168 edited model-condition evaluations were valid**, and all 42 processed candidate JPEG hashes agreed across the four scorers. Worst own-gallery cosine (maximum across seven conditions and four references) and valid nonmatch count are shown below; thresholds were SFace 0.515038, GhostFaceNet 0.343449, ArcFace 0.237600 and 0095 0.430598.

| Identity / arm | SFace cosine; NM | Ghost cosine; NM | ArcFace cosine; NM | 0095 cosine; NM |
| --- | ---: | ---: | ---: | ---: |
| 001 / A global | 0.378995; 7/7 | 0.176111; 7/7 | 0.466608; 0/7 | 0.318730; 7/7 |
| 001 / B balanced | 0.340187; 7/7 | 0.185707; 7/7 | 0.457518; 0/7 | 0.333550; 7/7 |
| 001 / C S/0095 | 0.355694; 7/7 | 0.337936; 7/7 | 0.508057; 0/7 | 0.063274; 7/7 |
| 003 / A global | 0.433299; 7/7 | 0.292154; 7/7 | 0.449066; 0/7 | 0.512765; 0/7 |
| 003 / B balanced | 0.313521; 7/7 | 0.457544; 0/7 | 0.523761; 0/7 | 0.546873; 0/7 |
| 003 / C S/0095 | 0.331400; 7/7 | 0.551096; 0/7 | 0.607925; 0/7 | 0.411799; 7/7 |

Predeclared ArcFace worst-cosine reductions of at least +0.05 on **both** identities failed: B−A was +0.009090/−0.074695 for 001/003, and C−B was −0.050539/−0.084163. No arm passed all seven SFace, GhostFaceNet and 0095 conditions on both identities. The largest paired processed face-RMS gap was 0.169897, under the 0.25 balance gate. The six native searches took 137.13 seconds excluding model loading, not a browser benchmark. **Retire this fixed H14 setting without expansion.** ArcFace and 0095 are development models; the four reserved final recognizers were untouched. No privacy, independent transfer or release claim follows.

## H15: including ArcFace in the gradient basis

The [SFace–ArcFace comparison](arcface-basis.md) used the same two development people, source-only references, dot carrier, distortion limits and 18-forward/17-update budget as H14. Both exact exported JPEGs passed preliminary root appearance screening, then all **56/56** edited model-condition evaluations were valid. SFace did not match either person in any of the seven conditions, but ArcFace matched both in **all seven**. The worst ArcFace gallery cosines were 0.419155 and 0.440566, above its frozen 0.237600 threshold. GhostFaceNet and 0095 also matched the second person throughout.

Adding ArcFace gradients reduced its scores relative to H14's balanced SFace/GhostFaceNet arm, but did not meet the predeclared nonmatch criterion. The maximum processed face-RMS difference from the matched H14 controls was 0.060, below 0.25. **Retire this fixed setting.** This conventional baseline is neither a novel transfer result nor release evidence. The linked note includes every model's results, controls, runtime and source/protocol fingerprints.

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
| Condition-specific alignment / both native recognizers | `09987614f0219835ba1f09c1e8fd8df75c4e748937297804d0313f2c7f79f30d` |
| ArcFace development calibration | `a186f3d285f4cff65d8c187a65f5929c19b13b86f8170583dee34dccab9bb56a` |
| Frozen condition-specific alignment / ArcFace | `a02863a43fb16b2eb9dee8415c4359fd26db4f27991f5fb47a8b13fbfc0e1ce29` |
| Relational dots / SFace and GhostFaceNet | `cee94666ccf6a9a4a25a717120635cc6497144431f54c9ca683d248db32989af` |
| Frozen relational dots / ArcFace | `3a51208d71ea943909ffe33e7aac5a46344f6370828042a7f06f23af9e57ab57` |
| Alignment transfer mechanism diagnostic | `03ffc152d10d70876a8f9ef2443ec72d4e84454adb9fd87e4d16d58d69b4fb15` |
| Released line-drawing model / SFace | `f105499fefc90b65a318d02c0cefc0ac11e5c6e8c5dcd8adb14c1ed8080a589f` |
| Momentum/input-diversity dots / SFace and GhostFaceNet | `c068309288696f14dd3247ba9bf267394fb7318b0c79f7c5603eab32c4956b74` |
| Frozen momentum/input-diversity dots / ArcFace | `b182b324c338044398dd5ddb7fc5f82c25df4e3410a4d26e5db055209e333434` |
| Sparse/inverse dot support / SFace and GhostFaceNet | `3f79d0945fd210abe9a977f61940cde620719b25bf1cc113089fbea9aa40a9cc` |
| Frozen sparse/inverse dot support / ArcFace | `c31565f1aede38168a72ea767e53e52f0c9d95dc4d7af446c5d36c157143a11c` |
| Contrast chimera executed renderer / pre-render protocol | `1e3f09b3d4644dde2c97d26bcdc91738658d7235109b0b8f17ea9785d06a472c` / `061ca9cc2ec7de71b03c9d47cf934f2c6df6beeae119984382bf506669a1a952` |
| Contrast chimera six-output freeze / three-row scoring freeze | `4ead5e5e4a3abe3f4758539859b2a4089677779cf8fdb78f6829ed43dcd8cdad` / `1c2d015ba93489f6a58f25cc4831c4e6383c78693d9280d0d8b641e129c1ce61` |
| Contrast chimera pre-score protocol / 63-row aggregate | `80151baec4e0edb1034c24da5b0683b2c2023fb3c6ffc148e94101f5442c42aa` / `b4d54ad38cabf6253fa0e30c45cd0f9e1db3970031fe1665a858026194917423` |
| Contrast chimera SFace / GhostFaceNet / ArcFace scores | `f20748e849853375749d7adf1d63e7ba0dfaa73b437a0305d55549e264edb7c5` / `b734779e299da0b586c1ea0c245aa6242d9488232ca91eb4c42288347fb7b366` / `3dae8e2cc7701820fd1849b36c49b1fd928a71edd4872b7777435e330669732e` |
| OpenVINO 0095 calibration | `f33a0e6fa03a4853013eea9fdda9c6c2d0c008a2f6655fd52064bb0187d1275a` |
| Frozen H7 dots / OpenVINO 0095 | `f68c320e51b1ceb5cc49a21867e60b3b706052e7bbb155397b1dc46b5dbaa60b` |
| Chimera dots executed source / optimization protocol / four-export freeze | `05d33fbbb90e2c499b83903753d4209bda29c6f0f2bd1aef15c481b2b27e881a` / `64682c04e154536d908cab051699dfa947deadab9036369ec7d34bd48f1b14da` / `95e4b04bfecf38d4d7c43f20d42ce5b9129d062f4ed4591b5989c40dc0aab0b7` |
| Chimera dots frozen-input manifest / pre-score protocol / disposition | `9f384685223c4e8ef27cbac995446c756aaa7886fbe46ab8c3d9da3aedbed363` / `11ac02989fabe54774f63ff5f45f8779f3748cd276971ab248017a67dbe4c91d` / `091e31a7b5167362bedcfbd39b2e86ff2dcbe6ee9c060790d8d7d40a59626956` |
| Chimera dots SFace / GhostFaceNet / ArcFace score JSON | `0c90e619efe1387021a0a1c13bed843ec80a832186681b34df7f60dfca1bb58b` / `b1009027784ca49915ba837e148c2585f5ee174accb895240cb8d32b7b0692ae` / `5096e90727b5602116ea913897f45e52d0d9bd7d52c4173f77f108a404fd2f5b` |
| H14 optimizer / 0095 alignment helper | `246a54398ec4449c52b4a18a773f7e5549c0565db0e96ae01aee839f94728ebe` / `bb3e9ac953f2b2e880d6ed47dd70d35147c41be4fcec704e0fef4241ff17a3ca` |
| H14 six-JPEG freeze / pre-score protocol / aggregate disposition | `512f8eb8976b3859b0eb236d36f10b17c092c3c76399e573b21edb3d9bf15773` / `2b0de2536888b1c930c1f8e1bbd038c891837148d972781cbb404deb5152eb5b` / `6b582c28149e8bdd2bece1a13e3461bc1488d5c0ec1114a2bac5d56edebc27ea` |
| H14 SFace / GhostFaceNet / ArcFace / OpenVINO 0095 score JSON | `22374bf3ea6876ab839272da9cbbbed0b1a4a713e98fbf3d9804f8accd7a0fb5` / `c0426bd7292786418527a581f5c09d9c4ae49f660531ca6924d8116275fdc697` / `a90dddd533f5a240e8d705d6af2d48d7d124ccb64ba67bc70c4053c1d9e752ae` / `8015a7480862ef4a4a92423d2484da6436bbb497425813af5f81ade881d86d2c` |

H15's frozen scoring protocol is `e05299268c75e95850e34c9920ecf88707e337776e34b27c3b2816124b54cc8a`; its checked aggregate is `580771938a936e693fcf5879ebcc799c530b638c2dca48be8b437e79023c8a92`. Full per-artifact fingerprints are in [the H15 note](arcface-basis.md).

The detailed reports contain local biometric artifacts and are not distributed. The aggregate findings above are the public record; no release success rate is claimed.
