# Development findings — 25 September 2026

**No method meets the release requirements.** These are small development experiments against SFace, which was used in earlier work. AdaFace, MagFace, EdgeFace and TransFace have not been evaluated. Independent human appearance review and real-phone testing are pending.

The [protocol](README.md) defines the frozen model, preprocessing and seven processing conditions. Each comparison uses the exact exported JPEG, fresh detection/alignment, and the original plus three separate clean references. A nonmatch requires every reference to fall below the calibrated threshold; a failed detection cannot count as a nonmatch. Repeated seeds, parameters and processing conditions are correlated observations, not additional people.

## Calibration

FRLL's separate calibration split contains 20 identities and 200 images. Native preprocessing produced 173 valid embeddings; 27 missing detections remained inconclusive. Of 19,900 possible pairs, 14,197 impostor and 681 genuine pairs were valid, for 74.76% pair coverage. At a target false-match rate of 0.001, the frozen cosine threshold is **0.5150383510**. Empirical false-match and false-nonmatch rates are 0.0009861 and 0.0969163. Leaving one identity out changes the estimated threshold from 0.488984 to 0.522429. This limited calibration is not a population guarantee.

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

The subsequent source-only ablation at RMS 16 selected exactly the same JPEG bytes and coefficients for both joint and whole-face arms on both people. It froze all selections before opening extra same-person reference images. The joint arm therefore retained one valid all-condition gallery nonmatch out of two eligible development people using only the source photo during generation; the whole-face control retained zero. Native execution took 207.8 seconds for four searches and final evaluation. A six-person confirmation uses the remaining prespecified screen identities without changing settings. This is open development, not independent release validation.

## Reproducibility

Runners, selection rules and model hashes are in this repository. Local run directories retain source snapshots, exact images, all reference scores, failures and timings outside Git. SHA-256 fingerprints of the completed detailed reports are:

| Report | SHA-256 |
| --- | --- |
| Combined artwork analysis | `bf21492168ab9ed5a64d6340d9669daa5a2a90c1aa1a6d25ab88ee2655657848` |
| Regional oracle results | `1b2bda72eca3b9fe1d784133345ec1dddd009a4b34201045790ecb9e72a406b4` |
| Graphic bottleneck scores | `474380712fcdd35b386b0c4b44457b1d0d8681695d8d78b2c7ed58835451b03d` |

The detailed reports contain local biometric artifacts and are not distributed. The aggregate findings above are the public record; no release success rate is claimed.
