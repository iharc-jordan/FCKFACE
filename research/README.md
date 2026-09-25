# Reproducible development research

These experiments are development evidence, not release validation. The accepted product and validation requirements live in [REQUIREMENTS.md](../REQUIREMENTS.md).

## Environment and data

Use Python 3.12 with `research/requirements.txt`; run unit tests from `research/` with `python -m unittest discover -s tests -v`. The measured native run used OpenCV 4.13.0, NumPy 2.5.3 and Pillow 12.3.0. Individual run manifests record exact versions and source hashes.

Optional gradient diagnostics use `research/requirements-gradient.txt` in a separate environment. `sface_gradient_reference.py` checks the cached official SFace ONNX against OpenCV on a deterministic synthetic input and checks the input gradient with finite differences. Pass `--model` and an external `--output` path. It does not evaluate faces or establish browser parity.

Optional [GhostFaceNet development evaluation](development-models.md) uses `research/requirements-ghostface.txt`. Its adapter loads the serialized author graph with legacy Keras; it shares the declared YuNet detector substitution but has its own author alignment, normalization and calibrated threshold. It is not a member of the reserved final panel.

The [completed findings](results.md) include negative cross-model transfer and rejected appearance settings. A third [official ArcFace development pipeline](arcface-development.md) exposed a transfer failure after dots succeeded on the two optimized models. [Portrait reproduction instructions](learned-portrait.md) pin the external author code and checkpoints used for the learned-drawing comparison. No images, embeddings or model weights are included in this source repository.

A fourth [OpenVINO 0095 development pipeline](openvino-development.md) uses Intel's pinned FP32 artifact and demo preprocessing with the declared YuNet substitution. Its separate calibration and four-export H7 diagnostic are complete; only one of two people passed all seven conditions. Its isolated environment uses OpenVINO 2025.4.1, OpenCV 4.13.0, NumPy 2.3.5 and Pillow. The model metadata grants Apache-2.0 for the artifacts, but browser conversion and a deployable method remain unverified. See [research attribution](NOTICE).

The compact model's [ONNX conversion and native gradient checks](openvino-onnx-conversion.md) passed on three synthetic inputs and two previously reviewed aligned face crops. These component checks do not establish browser performance or an effective editing method. The [chimera-aware dot comparison](chimera-dots.md) failed its recognition gate despite matched distortion and preliminary appearance acceptance; its detector failures remain inconclusive.

The subsequent [TensorFlow conversion](browser/openvino_tfjs_conversion.md) passed forward checks but missed the predeclared gradient tolerance on one synthetic case. That failure and its bounded numerical diagnosis are preserved; no browser adoption followed.

Acquire the consented [Face Research Lab London Set, version 5](https://doi.org/10.6084/m9.figshare.5047666.v5):

```powershell
python research/fckface_lab/datasets.py --root C:/path/outside/repository/FCKFACE-data
```

This checks upstream archive digests and writes an image inventory with hashes. The 102 identities are assigned once by `frll-identities-v1` to 60 development, 20 calibration and 22 held-out identities, before optimization. Ten views per identity provide separate clean references. Existing splits are never silently replaced. Dataset authors: Lisa DeBruine and Benedict Jones; images are CC BY 4.0. The source describes participant permission for original and altered images in research. Data remain outside Git despite the permissive license.

[FEI](https://fei.edu.br/~cet/facedatabase.html) is research-use data, not an application asset. Its source currently returns HTTP 403; acquisition and identity deduplication are pending.

A separate [Commons development acquisition inventory](commons-development-manifest.json) records 32 original photographs across eight source-labeled people, totaling 109,891,253 bytes. File pages, attribution, licenses and hashes are included; photographs and copied descriptions are not. The metadata selection was frozen before downloads and recognition scoring. All originals were verified and reviewed as full frames. Group scenes, profiles and small faces require explicit subject annotations; one selected file has no visible face, and another has conflicting event/date metadata. No recognition models have evaluated this set, its four-photo galleries cannot be assumed complete, and cross-dataset identity deduplication remains unresolved. These limitations must be reported rather than silently replacing difficult examples. Less controlled, eligible photographs are still required before broader real-world claims. Never copy research images or embeddings into the public application.

Download development weights from the official locations in [models.json](models.json) and verify SHA-256. Supply model paths explicitly to the runners. YuNet's model directory has an MIT license; SFace's declares Apache-2.0. These terms are separate from this project's license. No weights are distributed here. AdaFace, MagFace, EdgeFace and TransFace remain reserved and unmeasured.

## Native evaluation

Decode orientation and colour profiles to sRGB; detect with official YuNet at a maximum image side of 640 pixels, map coordinates back, then use OpenCV SFace's native `alignCrop` and `feature`. A uniquely selected face is required. Invalid, missing or ambiguous faces remain inconclusive. The original-resolution detector missed clear FRLL faces; that interrupted calibration was discarded before freezing results.

Run `python research/calibrate_sface.py --help` for paths. The calibration uses all 20 separate calibration identities and all ten views; the target false-match rate is 0.001. Missing embeddings reduce reported coverage and cannot become false nonmatches. Freeze the resulting threshold and pipeline before development scoring. Scores are cosines, not probabilities. Calibration pairs are correlated, so pair count alone overstates independent evidence; the output includes identity sensitivity and sampling limitations.

Evaluate the exact exported JPEG after fresh alignment in seven fixed conditions: Q95 4:4:4 export; Q85 and Q75 4:2:0; longest-side 960 resize; half-size downsample and restoration; central 90% crop; Gaussian blur radius 1. All transformed images are encoded and decoded before recognition. Every source is compared with its clean original and separate front/three-quarter photos. Clean controls establish eligibility before considering edits. Save all per-reference scores, missing-data reasons and exact bytes.

## First bounded screen

`run_screen.py` preselects the first eight sorted development identities, with neutral-front source and smiling-front plus both neutral three-quarter views as clean references. Seeds are 0 and 1; face-local RMS budgets are 4 and 8 channel levels. Render amplitudes are bounded by 64, and unattainable distortion targets are explicitly unmatched. A two-identity pilot catches pipeline errors before the full eight-identity run.

| Family | Candidate and matched controls | What would justify further work |
| --- | --- | --- |
| Landmark-relative artwork | Canonical multiscale dots/rings/lines versus equal-frequency image-anchored patterns and ordinary dots | Advantage survives image processing, seeds and identities at matched distortion |
| Joint regions | Coordinated eyes/cheeks/nose/lower-face versus each region individually | Joint effect exceeds regional controls; later optimization must also compare independently optimized regions and unrestricted whole-face search |
| Selective detail replacement | Graphic round-dot reconstruction versus ordinary Bayer screening and smoothing at the same coverage | Less matching after export without ambiguous identity or anatomy-like artifacts |

This first screen tests fixed renderings, not optimized attacks. It does not establish the full optimization hypotheses. Do not extend a failed grid indefinitely: propose a changed mechanism and budget. Root review includes complete photos and face crops; human acceptance still requires independent blinded reviewers and separate reference images. Novelty is unproven. [LowKey](https://arxiv.org/abs/2101.07922) already accounts for recognition pipelines in adversarial photo editing.

The two-identity pilot completed all 88 candidates and 616 comparisons, with zero gallery nonmatches. Only the artwork comparison advanced to the remaining six identities. Combined H1 evidence has 96 candidates and 672 valid comparisons, all still matched. The fitted canonical scale differs from the global control's scale, so a lower score cannot establish an anchoring mechanism. The skin-coloured graphic-halftone rendering was rejected as a product candidate because its appearance can resemble pigmentation.

Follow-up work tests two changed mechanisms: `optimize_regions.py` searches signed regional coefficients under a bounded query budget; `reconstruct_graphics.py` tests actual detail removal followed by opaque, non-skin-coloured graphics. The regional run has access to extra clean references during optimization and is therefore an **oracle diagnostic**, not a one-photo deployment method. A product candidate must generate from the single supplied photo, with other reference photos reserved for evaluation or offline development.

The additional Chrome discussion prompted the detail-bottleneck test and the single-query constraint check. Its suggestions are hypotheses, not new product requirements. Relevant prior work includes [personalized face masks](https://arxiv.org/abs/2605.19032), [alignment robustness](https://arxiv.org/abs/2407.14972), [low/mid-frequency perturbations](https://arxiv.org/abs/2206.09410), and [dynamic-gallery evaluation](https://arxiv.org/abs/2501.06533). None establishes that these FCKFACE experiments work or are novel.

Keep run settings, source/model/data hashes, exported JPEGs, all variants, timings, errors and measurements in an external output directory. Only sanitized aggregate findings belong in this repository. The held-out gate in `evaluation.py` implements the numerical aggregation; it cannot certify identity independence, artifact rights, human acceptance, sample adequacy or usability by itself.
