# FCKFACE project guidance

Read the global `C:/Users/JordanStevenson/.codex/REQUIREMENTS.md` and this repository's `REQUIREMENTS.md` before substantial work. Later explicit direction from Jordan takes precedence.

- FCKFACE is intended to let someone submit a photo and receive a modified photo that resists facial recognition. Treat that as a research goal until measured across relevant, independent models and image processing paths. Never describe an unvalidated edit as anonymous or unmatchable.
- The earlier Face Privacy Filter pilot lives separately in Jordan's `Downloads/Face Privacy Filter` repository. Its scripts and measurements are evidence, not production code or current product requirements. See `docs/prior-work.md` before resuming the research.
- Keep personal photos, reference galleries, face embeddings, downloaded model weights, generated images, experiment output, and secrets out of Git. Check tracked files before pushing.
- Preserve source originals. When research resumes, evaluate the exact exported image, keep detection or face-selection failures inconclusive, and distinguish development results from independent transfer tests.
- Keep the implementation small: React/TypeScript/Vite, browser-first processing, and a separate local research harness. Follow the release and appearance requirements in REQUIREMENTS.md; a research preview is not a validated public processor.
