/** Local inspection of exact YuNet ONNX output shapes and first development tensor. */
import { createRequire } from 'node:module'
import { readFile } from 'node:fs/promises'
import { join, sep } from 'node:path'
import { pathToFileURL } from 'node:url'
import { decodeYunet } from './decode_yunet.mjs'

const scratch = process.argv[2]
const model = process.argv[3]
if (!scratch || !model) throw new Error('Usage: node yunet_node_parity.mjs <scratch> <yunet.onnx>')
const require = createRequire(join(scratch, 'package.json'))
const ort = require('onnxruntime-web')
ort.env.logLevel = 'error'
ort.env.wasm.numThreads = 1
ort.env.wasm.wasmPaths = pathToFileURL(join(scratch, 'node_modules', 'onnxruntime-web', 'dist') + sep).href
const manifest = JSON.parse(await readFile(join(scratch, 'assets', 'yunet-manifest.json'), 'utf8'))
const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'], logSeverityLevel: 3 })
const cases = []
for (const item of manifest.cases) {
  const raw = await readFile(join(scratch, 'assets', item.assets.detector_blob.name))
  const input = new ort.Tensor('float32', new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)), item.detector_shape)
  const output = await session.run({ [session.inputNames[0]]: input })
  const faces = decodeYunet(output, ...item.padded_size)
  const errors = faces.length === 1 ? faces[0].map((value, index) => value - item.native_detection_frame[index]) : null
  cases.push({ identity: item.identity, nativeDetectionCount: 1, jsDetectionCount: faces.length,
    faces, maxAbsFrameDifference: errors ? Math.max(...errors.map(Math.abs)) : null, errors })
}
console.log(JSON.stringify({ inputNames: session.inputNames, outputNames: session.outputNames, cases }, null, 2))
await session.release()
