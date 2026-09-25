# FCKFACE

**Face Cloaking Kit for Feature Alteration and Cross-model Evaluation**

FCKFACE researches visible photo edits intended to make facial matching harder while preserving the person's identity to human viewers. The name is intentionally a joke; the effectiveness claim needs serious evidence.

## Status

The [public research site](https://fckface.vercel.app), local browser prototype and reproducible development evaluation harness are available. **No method has passed the protection or human-appearance requirements.** The public site contains a research status page, with no photo processor. A working image effect is not evidence of facial privacy.

The previous single-user research repository and its private evidence remain on Jordan's computer. [Prior work and handoff](docs/prior-work.md) identifies what was tried and where to find it. Personal photographs, embeddings, model weights, and generated results are excluded from this repository.

## Local browser prototype

Requires Node.js 24 or later. Install with `npm ci`, then run `npm run dev`. Select a JPEG, PNG or WebP, mark one face, process locally, compare the full image and crop, and download a new JPEG. The current dot effect tests the browser workflow only; it is not a selected protection method. Images stay on the device and source metadata is not copied to the export.

- `npm run build`: typecheck and build the public status page. A bundle check rejects accidentally included research processing.
- `npm run build:research`: build the local prototype into `dist-research/`. Do not deploy this as a validated processor.
- `npm run test:browser`: exercise selection, orientation, download, metadata removal, cancellation, repeated use and network behavior with Playwright. Install its Chromium with `npx playwright install chromium --only-shell` first.

Desktop browser emulation does not establish real iPhone or Android support. Processing currently uses a Web Worker; model inference and optimization feasibility are separate experiments.

## Research

The [research protocol](research/README.md) documents data acquisition, identity splits, calibrated native evaluation, hypotheses, controls and reproducible commands. [Development findings](research/results.md) include unsuccessful approaches, coverage and limitations. Keep datasets, photos, embeddings, model weights and generated outputs outside this repository. [Model provenance](research/models.json) separates development models from the reserved final panel.

Public processing requires at least 95% of eligible unseen photos to fail clean-gallery matching on at least three of four independent models across every required image-processing condition, plus separate human appearance and device usability validation. Detection, alignment and selection failures are inconclusive. See [the accepted requirements](REQUIREMENTS.md) for the complete scope.

Original project code is licensed under [Apache-2.0](LICENSE). Third-party datasets and model artifacts retain their own terms; this license does not grant rights to them.
