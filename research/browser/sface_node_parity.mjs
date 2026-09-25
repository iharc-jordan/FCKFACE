/** Local ONNX Runtime Web WASM parity probe; test assets stay outside Git. */
import { createRequire } from 'node:module'
import { readFile } from 'node:fs/promises'
import { resolve, join, sep } from 'node:path'
import { pathToFileURL } from 'node:url'
import { performance } from 'node:perf_hooks'
import { createHash } from 'node:crypto'

const [scratchArg, modelArg] = process.argv.slice(2)
if (!scratchArg || !modelArg) throw new Error('Usage: node sface_node_parity.mjs <scratch-dir> <model.onnx>')
const scratch = resolve(scratchArg)
const require = createRequire(join(scratch, 'package.json'))
const ort = require('onnxruntime-web')
const ortVersion = JSON.parse(await readFile(join(scratch, 'node_modules', 'onnxruntime-web', 'package.json'), 'utf8')).version
ort.env.logLevel = 'error'
ort.env.wasm.numThreads = 1
ort.env.wasm.wasmPaths = pathToFileURL(join(scratch, 'node_modules', 'onnxruntime-web', 'dist') + sep).href

const manifest = JSON.parse(await readFile(join(scratch, 'assets', 'manifest.json'), 'utf8'))
const modelBytes = await readFile(resolve(modelArg))
const modelSha256 = createHash('sha256').update(modelBytes).digest('hex')
if (modelSha256 !== manifest.recognizer_sha256) throw new Error('SFace model hash mismatch')
const initialRss = process.memoryUsage().rss
const started = performance.now()
const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ['wasm'], logSeverityLevel: 3 })
const loadMs = performance.now() - started
const rssAfterLoad = process.memoryUsage().rss
const inputName = session.inputNames[0]
const outputName = session.outputNames[0]
const rows = []
for (const item of manifest.cases) {
  const raw = await readFile(join(scratch, 'assets', `${item.label}.f32`))
  const bytes = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)
  const input = new ort.Tensor('float32', new Float32Array(bytes), item.tensor_shape)
  const samples = []
  let output
  for (let index = 0; index < 6; index++) {
    const mark = performance.now()
    output = (await session.run({ [inputName]: input }))[outputName]
    if (index > 0) samples.push(performance.now() - mark)
  }
  const values = output.data
  const reference = item.reference_feature
  const norm = vector => Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0))
  const cosine = values.reduce((sum, value, index) => sum + value * reference[index], 0) / (norm(values) * norm(reference))
  const maxAbs = values.reduce((maximum, value, index) => Math.max(maximum, Math.abs(value - reference[index])), 0)
  rows.push({ case: item.label, identity: item.identity, shape: output.dims, maxAbs, cosine,
    warmMs: samples, averageWarmMs: samples.reduce((a, b) => a + b, 0) / samples.length })
}
const rssBeforeRelease = process.memoryUsage().rss
await session.release()
console.log(JSON.stringify({ packageVersion: ortVersion,
  backend: 'onnxruntime-web/wasm', modelSha256,
  inputName, outputName, loadMs, initialRss, rssAfterLoad, rssBeforeRelease,
  rssAfterRelease: process.memoryUsage().rss, rows }, null, 2))
