# Prior Face Privacy Filter work

This is a setup handoff, not a new experiment or a claim of facial privacy effectiveness. The earlier Codex task is `01a0d5b8-f346-79f2-8c8d-ea44595f7fa6`. It stopped its research runs and handed over on September 25, 2026. Its separate local Git repository is in Jordan's `Downloads/Face Privacy Filter` folder on branch `codex/privacy-research`, at handoff commit `b208072`. That repository retains the pilot scripts, research protocol, model-source notes, and local experiment records. Private photographs, embeddings, downloaded weights, and generated results remain outside this FCKFACE repository.

## What was tried

- The pilot compared edited photos to the original and other reference photos using several local recognition models, notably SFace, ArcFace, FaceNet512, and GhostFaceNet. It tested exported JPEGs after changes such as resize, recompression, and alternate face alignment.
- Earlier approaches included small geometry changes, smooth and pixel-level colour changes, frequency and JPEG coefficient changes, relighting, generative reconstructions, and alignment-focused searches. Several stronger edits left vein-like or otherwise misleading changes to facial appearance; Jordan rejected those.
- Jordan preferred a clearly artificial regular-dot effect in the latest visual preview. A pilot varied dot strength by cell and compared it with a uniform dot grid at approximately matched distortion. The optimized version improved the pilot's heuristic score, but the native SFace checks still matched the edited photo to all 11 source conditions. The appearance preference is not a recognition result.
- A separate bounded RGB continuation improved some development scores but still left source matches. The local protocol notes that the models and reference photos used for selection are development evidence, not independent transfer validation.
- A later RGB continuation was interrupted and has only progress records, without final repeat checks or a selected result. A dot-continuation script remains an untracked, unreviewed draft in the old local repository. Neither is a completed result. The old task stopped its Python runs, and the requested final photos were not changed in this research round.

## Current conclusion and transfer boundary

No tested workflow has reliably prevented matching, and no novel contribution is established. The previous scripts are tailored to one private pilot, with local paths and experiment-specific assumptions. They have not been copied into this clean product repository. Preserve the prior Git history and local results for later review; select or generalize code only when Jordan resumes the work.

Start any later technical review with the old repository's `HANDOFF-FCKFACE.md`, `RESEARCH.md`, `MODEL-SOURCES.md`, `REQUIREMENTS.md`, and `work/task-state/pattern-free-pilot.md`, then inspect its current Git status and exact experiment results. The earlier project's local-only, no-UI scope has been superseded by FCKFACE's upload-and-result goal. Product architecture and inference location remain undecided.
