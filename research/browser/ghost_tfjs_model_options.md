# Browser model formats and artifact terms — 2026-09-25

Both development recognizers have now passed synthetic TensorFlow.js output and input-gradient checks in desktop Chrome. See the [SFace conversion record](sface_tfjs_conversion.md) and [GhostFaceNet record](ghost_tfjs_conversion.md). These are separate component checks, not a complete photo optimizer, phone benchmark or selected release method. The earlier conversion prospect is preserved in Git history; the tested formats below supersede it.

## SFace

The original OpenCV Zoo `face_recognition_sface_2021dec.onnx` is 38,696,353 bytes, SHA-256 `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`. Its runtime input is RGB NCHW `[1,3,112,112]` in 0–255, and its output has 128 values. It contains its own normalization. The standard converted TF.js GraphModel instead accepts NHWC `[1,112,112,3]` in the same RGB range; adding external normalization would be incorrect. GraphModel exposes input/output names `data` and `Identity`, although the artifact signature includes `:0` suffixes.

The [OpenCV Zoo SFace directory](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/README.md) affirmatively declares Apache-2.0 for all its files, including the ONNX. Its exact training dataset is not documented there. [Issue 313](https://github.com/opencv/opencv_zoo/issues/313) requests further provenance; that unanswered third-party question does not revoke the directory's license declaration. Model provenance, conversion correctness and effectiveness are separate matters. No model is bundled in the current public status site.

## GhostFaceNet

The browser model is an exact-architecture float32 clone of the serialized author H5, with the same weights and BatchNorm settings and with training-only regularizers removed. Native author H5 remains authoritative for development scoring. The converted clone's successful synthetic gradient checks do not make the different DeepFace reconstruction a valid substitute.

The [author repository](https://github.com/HamadYA/GhostFaceNets) has an [MIT software license](https://github.com/HamadYA/GhostFaceNets/blob/main/LICENSE) and offers the cached checkpoint in its [v1.2 release](https://github.com/HamadYA/GhostFaceNets/releases/tag/v1.2). The README identifies MS1MV3 training and links InsightFace's MS1M-RetinaFace dataset. The reviewed author pages do not contain a distinct checkpoint license or an express public browser redistribution/commercial-use grant. A release download and software license alone do not resolve that scope.

[InsightFace's license statement](https://github.com/deepinsight/insightface#license) restricts its training data and models trained with that data to noncommercial research while licensing code under MIT. This informs the provenance question but does not itself resolve the exact GhostFaceNet author's checkpoint terms. Converted shards retain the same weights, so conversion does not avoid that question. GhostFaceNet weights remain local; public distribution requires an affirmative rights basis.
