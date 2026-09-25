/** Local-only browser forward parity against native OpenCV SFace references. */
import * as ort from '/ort/ort.min.mjs'

const status = document.getElementById('status')
const result = document.getElementById('result')
const mark = () => performance.now()
const fetchBytes = async path => {
  const response = await fetch(path, { cache: 'no-store' })
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`)
  return new Uint8Array(await response.arrayBuffer())
}
const sha256 = async bytes => {
  const digest = await crypto.subtle.digest('SHA-256', bytes)
  return Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('')
}
const norm = vector => Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0))

try {
  ort.env.logLevel = 'error'
  ort.env.wasm.numThreads = 1
  ort.env.wasm.wasmPaths = `${location.origin}/ort/`
  const manifest = await (await fetch('/manifest.json', { cache: 'no-store' })).json()
  const model = await fetchBytes('/model.onnx')
  const modelHash = await sha256(model)
  if (modelHash !== manifest.recognizer_sha256) throw new Error('SFace model hash mismatch')
  const heapBefore = performance.memory?.usedJSHeapSize ?? null
  const start = mark()
  const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'], logSeverityLevel: 3 })
  const loadMs = mark() - start
  const [inputName] = session.inputNames
  const [outputName] = session.outputNames
  const rows = []
  for (const item of manifest.cases) {
    status.textContent = `Testing ${item.label}…`
    const raw = await fetchBytes(`/${item.label}.f32`)
    if (await sha256(raw) !== item.tensor_sha256) throw new Error(`${item.label} tensor hash mismatch`)
    const input = new ort.Tensor('float32', new Float32Array(raw.buffer), item.tensor_shape)
    const warmMs = []
    let output
    for (let index = 0; index < 6; index++) {
      const begin = mark()
      output = (await session.run({ [inputName]: input }))[outputName]
      if (index > 0) warmMs.push(mark() - begin)
    }
    const values = output.data
    const reference = item.reference_feature
    const cosine = values.reduce((sum, value, index) => sum + value * reference[index], 0) / (norm(values) * norm(reference))
    const maxAbs = values.reduce((maximum, value, index) => Math.max(maximum, Math.abs(value - reference[index])), 0)
    rows.push({ case: item.label, identity: item.identity, outputShape: output.dims,
      maxAbs, cosine, warmMs, averageWarmMs: warmMs.reduce((a, b) => a + b, 0) / warmMs.length })
  }
  await session.release()
  const report = {
    scope: 'Development forward-inference parity only; no privacy claim',
    userAgent: navigator.userAgent,
    backend: 'onnxruntime-web 1.30.0 / WASM single thread',
    opencvReferenceVersion: manifest.opencv_version,
    modelSha256: modelHash,
    inputName, outputName, loadMs,
    heapBefore, heapAfter: performance.memory?.usedJSHeapSize ?? null,
    rows,
  }
  window.__SFACE_PARITY_RESULT__ = report
  result.textContent = JSON.stringify(report, null, 2)
  status.textContent = 'Parity probe complete.'
} catch (error) {
  window.__SFACE_PARITY_ERROR__ = String(error?.stack ?? error)
  result.textContent = window.__SFACE_PARITY_ERROR__
  status.textContent = 'Parity probe failed.'
}
