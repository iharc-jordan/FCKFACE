/* Official OpenCV.js 4.13 image operations + ORT inference, local parity only.
 * The five-point template and YuNet decode are adapted from OpenCV 4.13.0:
 * https://github.com/opencv/opencv/blob/4.13.0/modules/objdetect/src/face_recognize.cpp
 * OpenCV contributors, Apache License 2.0; see repository LICENSE.
 */
importScripts('/opencv.js', '/ort/ort.min.js')

const ready = () => new Promise(resolve => {
  if (typeof cv.Mat === 'function') { resolve(); return }
  const prior = cv.onRuntimeInitialized
  cv.onRuntimeInitialized = () => { prior?.(); resolve() }
})
const fetchBytes = async path => {
  const response = await fetch(path, { cache: 'no-store' })
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`)
  return new Uint8Array(await response.arrayBuffer())
}
const hash = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
  value => value.toString(16).padStart(2, '0')).join('')
const norm = vector => Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0))
const progress = text => self.postMessage({ type: 'progress', text })
async function rgba(bytes) {
  const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/png' }),
    { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height)
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    ctx.drawImage(bitmap, 0, 0)
    return ctx.getImageData(0, 0, bitmap.width, bitmap.height)
  } finally { bitmap.close() }
}
function bgrMat(image) {
  const mat = new cv.Mat(image.height, image.width, cv.CV_8UC3)
  const target = mat.data
  const source = image.data
  for (let pixel = 0; pixel < image.width * image.height; pixel++) {
    target[3 * pixel] = source[4 * pixel + 2]
    target[3 * pixel + 1] = source[4 * pixel + 1]
    target[3 * pixel + 2] = source[4 * pixel]
  }
  return mat
}
function nchwBgr(mat) {
  const plane = mat.rows * mat.cols
  const output = new Float32Array(3 * plane)
  for (let pixel = 0; pixel < plane; pixel++) for (let channel = 0; channel < 3; channel++) {
    output[channel * plane + pixel] = mat.data[3 * pixel + channel]
  }
  return output
}
function nchwRgb(mat) {
  const plane = mat.rows * mat.cols
  const output = new Float32Array(3 * plane)
  for (let pixel = 0; pixel < plane; pixel++) {
    output[pixel] = mat.data[3 * pixel + 2]
    output[plane + pixel] = mat.data[3 * pixel + 1]
    output[2 * plane + pixel] = mat.data[3 * pixel]
  }
  return output
}
function similarity(points) {
  const target = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]]
  const srcX = points.reduce((sum, point) => sum + point[0], 0) / 5
  const srcY = points.reduce((sum, point) => sum + point[1], 0) / 5
  const dstX = target.reduce((sum, point) => sum + point[0], 0) / 5
  const dstY = target.reduce((sum, point) => sum + point[1], 0) / 5
  let dot = 0; let cross = 0; let denominator = 0
  for (let index = 0; index < 5; index++) {
    const x = points[index][0] - srcX; const y = points[index][1] - srcY
    const u = target[index][0] - dstX; const v = target[index][1] - dstY
    dot += x * u + y * v
    cross += x * v - y * u
    denominator += x * x + y * y
  }
  if (denominator <= 0) throw new Error('Degenerate landmarks')
  const a = dot / denominator; const b = cross / denominator
  const tx = dstX - a * srcX + b * srcY
  const ty = dstY - b * srcX - a * srcY
  return [a, -b, tx, b, a, ty]
}

async function run() {
  progress('Initializing official OpenCV.js 4.13…')
  await ready()
  const { decodeYunet } = await import('/decode_yunet.mjs')
  ort.env.logLevel = 'error'
  ort.env.wasm.numThreads = 1
  ort.env.wasm.wasmPaths = `${location.origin}/ort/`
  const manifest = await (await fetch('/yunet-manifest.json', { cache: 'no-store' })).json()
  const yunetBytes = await fetchBytes('/yunet-model.onnx')
  const sfaceBytes = await fetchBytes('/model.onnx')
  if (await hash(yunetBytes) !== manifest.detector_sha256 || await hash(sfaceBytes) !== manifest.recognizer_sha256) throw new Error('Model hash mismatch')
  progress('Loading ONNX Runtime sessions…')
  const loadStart = performance.now()
  const yunet = await ort.InferenceSession.create(yunetBytes, { executionProviders: ['wasm'], logSeverityLevel: 3 })
  let sface
  try {
    sface = await ort.InferenceSession.create(sfaceBytes, { executionProviders: ['wasm'], logSeverityLevel: 3 })
    const loadMs = performance.now() - loadStart
    const rows = []
    for (const item of manifest.cases) {
      progress(`OpenCV.js resize, detection and warp: ${item.label}…`)
      const tensorBytes = await fetchBytes(`/${item.assets.detector_blob.name}`)
      const sourceBytes = await fetchBytes(`/${item.assets.source_png.name}`)
      const alignedBytes = await fetchBytes(`/${item.assets.aligned_png.name}`)
      for (const [name, bytes] of [['detector_blob', tensorBytes], ['source_png', sourceBytes], ['aligned_png', alignedBytes]]) {
        if (await hash(bytes) !== item.assets[name].sha256) throw new Error(`${item.label} ${name} hash mismatch`)
      }
      const sourceImage = await rgba(sourceBytes)
      const nativeImage = await rgba(alignedBytes)
      const source = bgrMat(sourceImage)
      const resized = new cv.Mat()
      const aligned = new cv.Mat()
      const matrix = new cv.Mat(2, 3, cv.CV_64FC1)
      try {
        cv.resize(source, resized, new cv.Size(...item.detector_size), 0, 0, cv.INTER_AREA)
        const detectorInput = nchwBgr(resized)
        const nativeInput = new Float32Array(tensorBytes.buffer)
        let resizeSum = 0; let resizeMax = 0; let resizeExact = 0
        for (let index = 0; index < nativeInput.length; index++) {
          const difference = Math.abs(detectorInput[index] - nativeInput[index])
          resizeSum += difference; resizeMax = Math.max(resizeMax, difference)
          if (difference === 0) resizeExact++
        }
        const detectStart = performance.now()
        const output = await yunet.run({ [yunet.inputNames[0]]: new ort.Tensor('float32', detectorInput, item.detector_shape) })
        const faces = decodeYunet(output, ...item.padded_size)
        const detectMs = performance.now() - detectStart
        if (faces.length !== 1) throw new Error(`${item.label}: expected one detection, got ${faces.length}`)
        const face = faces[0]
        const frameMaxAbs = Math.max(...face.map((value, index) => Math.abs(value - item.native_detection_frame[index])))
        const [originalW, originalH] = item.original_size
        const [detectorW, detectorH] = item.detector_size
        const mapped = face.map((value, index) => index < 14
          ? Math.fround(value * (index % 2 === 0 ? originalW / detectorW : originalH / detectorH)) : value)
        const mappedMaxAbs = Math.max(...mapped.map((value, index) => Math.abs(value - item.native_detection_mapped[index])))
        const points = Array.from({ length: 5 }, (_, index) => mapped.slice(4 + 2 * index, 6 + 2 * index))
        matrix.data64F.set(similarity(points))
        cv.warpAffine(source, aligned, matrix, new cv.Size(112, 112), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0))
        let pixelSum = 0; let pixelMax = 0; let pixelExact = 0
        for (let pixel = 0; pixel < 112 * 112; pixel++) for (let channel = 0; channel < 3; channel++) {
          const difference = Math.abs(aligned.data[3 * pixel + channel] - nativeImage.data[4 * pixel + 2 - channel])
          pixelSum += difference; pixelMax = Math.max(pixelMax, difference)
          if (difference === 0) pixelExact++
        }
        const featureStart = performance.now()
        const feature = (await sface.run({ [sface.inputNames[0]]: new ort.Tensor('float32', nchwRgb(aligned), [1, 3, 112, 112]) }))[sface.outputNames[0]].data
        const featureMs = performance.now() - featureStart
        const native = item.native_feature
        const featureMaxAbs = Math.max(...feature.map((value, index) => Math.abs(value - native[index])))
        const featureCosine = feature.reduce((sum, value, index) => sum + value * native[index], 0) / (norm(feature) * norm(native))
        rows.push({ label: item.label, identity: item.identity, resizeMeanAbs: resizeSum / nativeInput.length,
          resizeMaxAbs: resizeMax, resizeExactFraction: resizeExact / nativeInput.length,
          detectionCount: faces.length, frameMaxAbs, mappedMaxAbs, detectMs,
          pixelMeanAbs: pixelSum / (112 * 112 * 3), pixelMaxAbs: pixelMax,
          pixelExactFraction: pixelExact / (112 * 112 * 3), featureMaxAbs, featureCosine, featureMs })
      } finally {
        matrix.delete(); aligned.delete(); resized.delete(); source.delete()
      }
    }
    return { scope: 'Development detector/alignment parity only; no privacy claim',
      opencvJsSource: 'Official docs.opencv.org/4.13.0/opencv.js local copy',
      nativeOpenCvVersion: manifest.opencv_version, detectorSha256: manifest.detector_sha256,
      recognizerSha256: manifest.recognizer_sha256, userAgent: navigator.userAgent, loadMs, rows }
  } finally {
    await sface?.release(); await yunet.release()
  }
}
run().then(report => self.postMessage({ type: 'result', report }))
  .catch(error => self.postMessage({ type: 'error', error: String(error?.stack ?? error) }))
  .finally(() => self.close())
