# Held-out recognizer preparation (2026-09-25)

This records model and input choices **before** final-panel evaluation. No held-out identity, image, embedding, model output, or threshold was inspected or calculated in this preparation. The manifest is [`models.json`](models.json); these recognizers must not guide method development or candidate selection.

| Model | Predeclared primary checkpoint | Input path in the pinned author code | Repository / artifact rights |
| --- | --- | --- | --- |
| AdaFace | R100/MS1MV2 `adaface_ir101_ms1mv2.ckpt` | MTCNN-aligned 112×112 RGB crop, converted to BGR; CHW float32 `(pixel/255 − 0.5)/0.5` | Code MIT; checkpoint linked by author, separate weight/data terms unverified. |
| MagFace | DDP iResNet100/MS1MV2 `magface_epoch_00025.pth` | Five-point 112×112 alignment; `cv2.imread` BGR; `ToTensor` gives `[0,1]`; `Normalize(0,1)` leaves that range intact. Author feature path takes the unflipped input. | Code Apache-2.0; separate checkpoint/data terms unverified. |
| EdgeFace | EdgeFace-S γ=0.5 `edgeface_s_gamma_05.pt` | MTCNN-aligned 112×112 RGB crop; `ToTensor`, then per-channel Normalize(0.5,0.5) to `[-1,1]`. | Code BSD-3-Clause; checkpoint is in author repository, separate model/data rights unverified. |
| TransFace | TransFace-S/MS1MV2 `ms1mv2_model_TransFace_S.pt` | `inference.py` reads BGR, resizes 112×112, converts to RGB, then CHW float32 `(pixel/255 − 0.5)/0.5`. Its example assumes a face crop; the README's separate ModelScope path uses RetinaFace, largest-face selection, and alignment. | No license located in pinned repository root; code and checkpoint rights unresolved. |

The precise author revisions, checkpoint links, and per-model code links are frozen in `models.json`. The [AdaFace README](https://github.com/mk-minchul/AdaFace/blob/c60eaa786a42c03444f3df7096dbaf9d57ae010d/README.md) notes that [CVLFace](https://github.com/mk-minchul/CVLFace) is the newer official implementation; the choice above deliberately uses the original AdaFace repository and its advertised R100/MS1MV2 release. The [MagFace model zoo and alignment instructions](https://github.com/IrvingMeng/MagFace/blob/99bae614ac2643b9694bf18e0c5645272ff6acfa/README.md), [EdgeFace release instructions](https://github.com/otroshi/edgeface/blob/ce86851cfc37979a9cd2558598d0e9bc592cbba3/README.md), and [TransFace release instructions](https://github.com/DanJun6737/TransFace/blob/fb6cd56d04c7ea525e328219f5d5066adac03ada/README.md) are the primary selection sources. No checkpoint was selected by observed performance on final identities.

All four official checkpoints are **outside Git**, in an external `heldout-models/` directory. Author-hosted files were retrieved using their pinned links, without substituting mirrors or alternate checkpoints. Each file was hashed, never loaded:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `adaface_ir101_ms1mv2.ckpt` | 436,798,739 | `4a26839460d8e1a5e8a13a0d7968e919d464f482aa0f8350f9e2cec84fe37481` |
| `magface_epoch_00025.pth` | 283,040,192 | `cfeba792dada6f1f30d1e118aff077d493dd95dd76c77c30f57f90fd0164ad58` |
| `edgeface_s_gamma_05.pt` | 14,695,737 | `dc59abda2e8580399fd115a1eeb07e1f21156196db604b884407bcf0f17efb07` |
| `ms1mv2_model_TransFace_S.pt` | 346,705,361 | `4feccc6a0426c775a52a2e2d8750c9c1d04425f0cb604bd6b2bea24de458b8ac` |

Total on disk is 1,081,240,029 bytes. None of these files belongs in the repository or a public bundle.

Before final evaluation, fix detector, landmark source, alignment, handling of multiple faces, and gallery aggregation per recognizer; confirm loaded checkpoint/architecture compatibility; calibrate each model's threshold on calibration identities; and freeze these rules and the processing conditions. In particular, TransFace's exact checkpoint-specific detection/alignment path is unresolved. The original inference snippets establish pixel transforms but do not alone establish operating thresholds. A missing detection or alignment is inconclusive, not a failed match. No checkpoint or author preprocessing has yet been run against the final split.
