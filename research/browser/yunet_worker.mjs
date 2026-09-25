/** Local-only cancellable YuNet -> Canvas align -> SFace feasibility worker. */
import * as ort from '/ort/ort.min.mjs'
import { decodeYunet } from '/decode_yunet.mjs'

const fetchBytes = async path => {
  const response = await fetch(path, { cache: 'no-store' })
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`)
  return new Uint8Array(await response.arrayBuffer())
}
const sha256 = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
  value => value.toString(16).padStart(2, '0')).join('')
const norm = vector => Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0))
const progress = text => self.postMessage({ type: 'progress', text })

// Five-point template adapted from OpenCV 4.13.0 FaceRecognizerSF
// (Apache-2.0): https://github.com/opencv/opencv/blob/4.13.0/modules/objdetect/src/face_recognize.cpp
// Least-squares 2D similarity is the full-rank solution of its Umeyama fit.
function similarity(landmarks) {
  const target = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]]
  const srcX = landmarks.reduce((sum, point) => sum + point[0], 0) / 5
  const srcY = landmarks.reduce((sum, point) => sum + point[1], 0) / 5
  const dstX = target.reduce((sum, point) => sum + point[0], 0) / 5
  const dstY = target.reduce((sum, point) => sum + point[1], 0) / 5
  let dot = 0; let cross = 0; let denominator = 0
  for (let index = 0; index < 5; index++) {
    const x = landmarks[index][0] - srcX
    const y = landmarks[index][1] - srcY
    const u = target[index][0] - dstX
    const v = target[index][1] - dstY
    dot += x * u + y * v
    cross += x * v - y * u
    denominator += x * x + y * y
  }
  if (denominator <= 0) throw new Error('Degenerate landmarks')
  const a = dot / denominator
  const b = cross / denominator
  return [a, b, -b, a, dstX - a * srcX + b * srcY, dstY - b * srcX - a * srcY]
}

async function bitmapFromBytes(bytes) {
  return createImageBitmap(new Blob([bytes], { type: 'image/png' }),
    { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
}

function pixelsFromBitmap(bitmap, transform = null) {
  const canvas = new OffscreenCanvas(112, 112)
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'low'
  if (transform) ctx.setTransform(...transform)
  ctx.drawImage(bitmap, 0, 0)
  return ctx.getImageData(0, 0, 112, 112).data
}

function blobFromRgba(pixels) {
  const plane = 112 * 112
  const blob = new Float32Array(3 * plane)
  for (let pixel = 0; pixel < plane; pixel++) {
    blob[pixel] = pixels[4 * pixel]
    blob[plane + pixel] = pixels[4 * pixel + 1]
    blob[2 * plane + pixel] = pixels[4 * pixel + 2]
  }
  return blob
}

async function run() {
  ort.env.logLevel = 'error'
  ort.env.wasm.numThreads = 1
  ort.env.wasm.wasmPaths = `${location.origin}/ort/`
  const manifest = await (await fetch('/yunet-manifest.json', { cache: 'no-store' })).json()
  const yunetBytes = await fetchBytes('/yunet-model.onnx')
  const sfaceBytes = await fetchBytes('/model.onnx')
  if (await sha256(yunetBytes) !== manifest.detector_sha256 || await sha256(sfaceBytes) !== manifest.recognizer_sha256) {
    throw new Error('Model hash mismatch')
  }
  progress('Loading YuNet and SFace WASM sessions…')
  const loadStart = performance.now()
  const yunet = await ort.InferenceSession.create(yunetBytes, { executionProviders: ['wasm'], logSeverityLevel: 3 })
  let sface
  try {
    sface = await ort.InferenceSession.create(sfaceBytes, { executionProviders: ['wasm'], logSeverityLevel: 3 })
    const loadMs = performance.now() - loadStart
    const rows = []
    for (const item of manifest.cases) {
      progress(`Detecting ${item.label}…`)
      const tensorBytes = await fetchBytes(`/${item.assets.detector_blob.name}`)
      const sourceBytes = await fetchBytes(`/${item.assets.source_png.name}`)
      const alignedBytes = await fetchBytes(`/${item.assets.aligned_png.name}`)
      for (const [name, bytes] of [['detector_blob', tensorBytes], ['source_png', sourceBytes], ['aligned_png', alignedBytes]]) {
        if (await sha256(bytes) !== item.assets[name].sha256) throw new Error(`${item.label} ${name} hash mismatch`)
      }
      const input = new ort.Tensor('float32', new Float32Array(tensorBytes.buffer), item.detector_shape)
      const detectStart = performance.now()
      const output = await yunet.run({ [yunet.inputNames[0]]: input })
      const faces = decodeYunet(output, ...item.padded_size)
      const detectMs = performance.now() - detectStart
      if (faces.length !== 1) throw new Error(`${item.label}: expected one face, got ${faces.length}`)
      const face = faces[0]
      const frameMaxAbs = Math.max(...face.map((value, index) => Math.abs(value - item.native_detection_frame[index])))
      const [originalW, originalH] = item.original_size
      const [detectorW, detectorH] = item.detector_size
      const mapped = face.map((value, index) => index < 14
        ? Math.fround(value * (index % 2 === 0 ? originalW / detectorW : originalH / detectorH)) : value)
      const mappedMaxAbs = Math.max(...mapped.map((value, index) => Math.abs(value - item.native_detection_mapped[index])))
      const points = Array.from({ length: 5 }, (_, index) => mapped.slice(4 + 2 * index, 6 + 2 * index))
      progress(`Aligning ${item.label} in Canvas…`)
      const source = await bitmapFromBytes(sourceBytes)
      const reference = await bitmapFromBytes(alignedBytes)
      let canvasPixels; let nativePixels
      try {
        canvasPixels = pixelsFromBitmap(source, similarity(points))
        nativePixels = pixelsFromBitmap(reference)
      } finally {
        source.close(); reference.close()
      }
      let pixelAbsSum = 0; let pixelMaxAbs = 0; let exact = 0
      for (let index = 0; index < 112 * 112; index++) for (let channel = 0; channel < 3; channel++) {
        const difference = Math.abs(canvasPixels[4 * index + channel] - nativePixels[4 * index + channel])
        pixelAbsSum += difference
        pixelMaxAbs = Math.max(pixelMaxAbs, difference)
        if (difference === 0) exact++
      }
      const sfaceInput = new ort.Tensor('float32', blobFromRgba(canvasPixels), [1, 3, 112, 112])
      const featureStart = performance.now()
      const feature = (await sface.run({ [sface.inputNames[0]]: sfaceInput }))[sface.outputNames[0]].data
      const featureMs = performance.now() - featureStart
      const native = item.native_feature
      const featureMaxAbs = Math.max(...feature.map((value, index) => Math.abs(value - native[index])))
      const featureCosine = feature.reduce((sum, value, index) => sum + value * native[index], 0) / (norm(feature) * norm(native))
      rows.push({ label: item.label, identity: item.identity, detectMs, featureMs,
        detectionCount: faces.length, frameMaxAbs, mappedMaxAbs, mappedFace: mapped,
        pixelMeanAbs: pixelAbsSum / (112 * 112 * 3), pixelMaxAbs,
        pixelExactFraction: exact / (112 * 112 * 3), featureMaxAbs, featureCosine })
    }
    return { scope: 'Development detection/alignment parity only; no privacy claim',
      userAgent: navigator.userAgent, opencvVersion: manifest.opencv_version,
      detectorSha256: manifest.detector_sha256, recognizerSha256: manifest.recognizer_sha256,
      backend: 'onnxruntime-web 1.30.0 / WASM single thread', loadMs, rows }
  } finally {
    await sface?.release()
    await yunet.release()
  }
}

run().then(report => self.postMessage({ type: 'result', report }))
  .catch(error => self.postMessage({ type: 'error', error: String(error?.stack ?? error) }))
  .finally(() => self.close())
