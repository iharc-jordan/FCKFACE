# OpenVINO 0095 standard TensorFlow.js conversion feasibility

**Status: standard conversion and native TensorFlow forward parity passed, but the predeclared input-gradient parity gate failed on one of five cases. Browser execution remains untested.** This is development-model preparation, not adoption by the public app or evidence of facial privacy. The gradient component check used three synthetic tensors and two previously reviewed development face crops. No gallery, reserved final recognizer or browser was used.

The input is the previously verified ONNX conversion of Intel's official [`face-reidentification-retail-0095`](https://github.com/openvinotoolkit/open_model_zoo/blob/a6946b6d6ce42cbf4278df20275fab199655fc7d/models/intel/face-reidentification-retail-0095/model.yml) FP32 model, whose artifact terms point to [Apache-2.0](https://github.com/openvinotoolkit/open_model_zoo/blob/a6946b6d6ce42cbf4278df20275fab199655fc7d/LICENSE). That ONNX has opset 20, 4,466,620 bytes, SHA-256 `08c79d93c175877bb1f6591b9a72d4971a4c82fa03640589f762393d2d383963`; its OpenVINO-forward parity was established separately. The original has one float32 **BGR** input `[1,3,128,128]` in the range 0–255 and one raw 256-value output. The same frozen three synthetic inputs and OpenVINO raw outputs are in private `synthetic-reference.npz`, SHA-256 `94f95e7603b638b63ed65bdbc6984acfc46a846f67ce06416fbc4112816fa8f2`.

The ONNX and references were **copied as ordinary private files**, never linked, into `/home/jordan/fckface-0095-converter-v1/inputs/`. The existing read-only WSL converter environments were reused without installs or patches: onnx2tf 2.6.9 with TensorFlow 2.21.0 and TensorFlow.js converter 4.22.0 with TensorFlow 2.19.0, Python 3.12.3. Both jobs were CPU-limited to two threads. The standard route was:

```bash
onnx2tf -i inputs/model.onnx -o saved-model -tb tf_converter -osd
tensorflowjs_converter --input_format=tf_saved_model \
  --output_format=tfjs_graph_model --signature_name=serving_default \
  saved-model tfjs-graph
```

`onnx2tf` normalized **only the task-owned input copy** in place (its subsequent SHA-256 is `85370f44f4854433a5ac2d0db2f5d0f5b2967852197074f8e3d48089aa096981`). The authoritative ONNX archive was hash-checked afterward and remained `08c79d93…3963`. The SavedModel exposes float32 `[1,128,128,3]` and `[1,1,1,256]`. [openvino_tfjs_verify.py](openvino_tfjs_verify.py) transposed each frozen BGR NCHW input to BGR NHWC, without channel reversal or rescaling, and compared the raw converted TensorFlow output to the frozen OpenVINO output:

| Synthetic input | Maximum raw absolute difference | Unit cosine | Pass limits |
| --- | ---: | ---: | --- |
| Range | 1.729×10⁻⁶ | 0.9999999999988 | Yes |
| Flat | 3.278×10⁻⁶ | 0.9999999999970 | Yes |
| Random | 3.904×10⁻⁶ | 0.9999999999986 | Yes |

The prespecified limits were raw maximum difference ≤0.001 and cosine ≥0.99999 on **all three** inputs. The native SavedModel passed. The unmodified TensorFlow.js converter then produced a `graph-model`: `model.json` plus two float32 weight shards, 4,654,101 bytes total. Its signature records input `0` / `unknown:0`, float32 `[1,128,128,3]`, and output `output_0` / `Identity:0`, `[1,1,1,256]`. Those graph names are artifact metadata; actual JavaScript API names must be checked in a later browser probe. No browser forward or gradient parity is claimed here.

All converted assets, logs, the copied ONNX, frozen synthetic references and generated native parity report remain private under `/home/jordan/fckface-0095-converter-v1/`. The verifier source is [openvino_tfjs_verify.py](openvino_tfjs_verify.py). SHA-256 fingerprints:

| Artifact | SHA-256 |
| --- | --- |
| `saved-model/saved_model.pb` | `5f1e86f68f811a0b482058ad204cc9bcb128ef4f569c152d3ee31dfa1e77ad6b` |
| Native synthetic parity JSON | `0b626aef59ce1dda28de68de1e148906674d5e12b2549ae92aeb9439143f4330` |
| TF.js `model.json` | `a1881786ef6c45022707739cd3222dfbace0fdb82a9e15f927842515fe1cddcb` |
| Weight shard 1 / 2 | `60f2c0a532b94a7f0ed29336309513bec8e058f08f3e05814b38ee0a51c14e5b` / `15d1bb0c255639db2744c2b7fa7cda0e44f6b93dddf69d7e6becb5cdbdfb5a6c` |
| Conversion log / TF.js export log | `1c31d0940a05110b6e5fb496c633a4ac40f6b2c2b7ce72ad23a2d61a4640e79d` / `03e4f5b2019bc6d2429bbf04ef66f64c26f1481f7fa5a22e21fc08ac4d1f38ed` |
| Complete private conversion report | `0b75fa18616cb2a7ba3fada521f7132fc750859ac9d7fcbff21c94e732b750d1` |

The native TensorFlow `GradientTape` check used the fixed seed-7095 unit vector and scalar `dot(L2_normalize(raw256), target)` on the same three synthetic tensors and two previously reviewed 128-pixel BGR development crops. Its protocol was frozen before execution (SHA-256 `e848744010daaa2afb6f40acb55b6edb030f373cd126368c700c6fbcd789cff8`). All five forward comparisons passed raw maximum difference ≤0.001 and unit cosine ≥0.99999; all gradients were finite, nonzero and within maximum absolute difference 1×10⁻⁵ of frozen Torch gradients. Four of five also passed the required full-gradient cosine ≥0.99999. The synthetic random case measured **0.9999683289044792**, so the complete gradient gate **failed**. The other gradient cosines were range 0.9999999999991762, flat 0.9999977563072706, frll-024 0.9999999981312913, and frll-036 0.9999997843660497. The default-backend report SHA-256 is `fddde51575af1da7ea175cd652fcbedd1f52194537fb044ecb0fa8a594bc409f`; the [gradient verifier](openvino_tfjs_gradient_verify.py) source SHA-256 is `cdd156f3a4b7eadc16f16879826ccead526e835e810d7a8b6f19b05e9fda9804`.

One bounded diagnosis reran the unchanged five-case protocol in a fresh process with `TF_ENABLE_ONEDNN_OPTS=0` set before importing TensorFlow. All measured differences and cosines remained unchanged, including the random-case failure; report SHA-256 `a8035f0d5a99232f507997709cff482cd28e91676e9015debef20e229650a82f`. At the three largest interior random-case gradient discrepancies, central differences of the original ONNX Runtime and converted TensorFlow scalar were compared at fixed perturbations 0.01 and 0.1. The smaller perturbation had substantial float32 numerical noise; the larger brought both finite differences near their analytic gradients but did not isolate a different local derivative. The cause of the small directional difference remains unresolved. Diagnosis report SHA-256 is `5d2ebc56d02041aa3f565185f417f1aa64de6d8561b5fb7213ade7dfee143348`; [diagnosis source](openvino_tfjs_fd_diagnosis.py) SHA-256 is `6a4ad1be5f804e1d1616be711ec974dcd277473d9d97b39ac02b07f574ed77d4`. All four reports/protocol and the diagnosis source snapshot are private under `C:/Users/JordanStevenson/Downloads/FCKFACE-data/runs/openvino-0095-tfjs-v1/`.

The executed diagnosis source snapshot ends with an extra blank line. The published source omits that trailing whitespace and has SHA-256 `e9e5b00aa2715f8d9fecbd8f97e8bc0fe0f736faa4977d554c9ec96d00a06ec1`; its Python content otherwise matches the executed `6a4ad1be…77d4` snapshot. The original report and snapshot were preserved unchanged.

No browser probe, browser deployment, or further inference followed the failed gate. Browser forward and gradient behavior, device performance and use in an effective method remain unverified; the converted model has not been adopted. The official model card does not identify its training dataset.
