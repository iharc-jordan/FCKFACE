/* Local-only, two-model WebGL parity on two permitted development crops. */
importScripts('/tfjs/tf.min.js')

const progress = text => postMessage({ type: 'progress', text })
const limits = Object.freeze({
  sfaceRawMax: 1e-3, sfaceUnitCosineMin: .99999, sfaceGradientCosineMin: .99,
  ghostRawMax: 1e-3, ghostAuthorCosineMin: .999, ghostAuthorUnitMax: .02,
  ghostGradientCosineMin: .99,
})
function maxDifference(a, b) {
  if (a.length !== b.length) return Infinity
  let value = 0
  for (let i = 0; i < a.length; i++) value = Math.max(value, Math.abs(a[i] - b[i]))
  return value
}
function cosine(a, b) {
  if (a.length !== b.length) return NaN
  let dot = 0, aa = 0, bb = 0
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; aa += a[i] ** 2; bb += b[i] ** 2 }
  return dot / Math.sqrt(aa * bb)
}
function unit(a) {
  const norm = Math.sqrt(a.reduce((sum, value) => sum + value * value, 0))
  return a.map(value => value / norm)
}
async function sha256(bytes) {
  const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))
  return Array.from(hash, byte => byte.toString(16).padStart(2, '0')).join('')
}
async function json(url) {
  const response = await fetch(url, { cache: 'no-store' })
  if (!response.ok) throw Error(`Fetch failed: ${url}`)
  return response.json()
}
async function cropRgb(asset) {
  const response = await fetch(`/crops/${asset.name}`, { cache: 'no-store' })
  if (!response.ok) throw Error(`Crop fetch failed: ${asset.name}`)
  const png = new Uint8Array(await response.arrayBuffer())
  if (await sha256(png) !== asset.png_sha256) throw Error(`PNG checksum mismatch: ${asset.name}`)
  const bitmap = await createImageBitmap(new Blob([png], { type: 'image/png' }),
    { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
  try {
    if (bitmap.width !== 112 || bitmap.height !== 112) throw Error(`Crop dimensions changed: ${asset.name}`)
    const canvas = new OffscreenCanvas(112, 112)
    const context = canvas.getContext('2d', { willReadFrequently: true, colorSpace: 'srgb' })
    if (!context) throw Error('OffscreenCanvas 2D unavailable')
    context.drawImage(bitmap, 0, 0)
    const rgba = context.getImageData(0, 0, 112, 112).data
    const rgb = new Uint8Array(112 * 112 * 3)
    for (let i = 0; i < 112 * 112; i++) {
      if (rgba[4 * i + 3] !== 255) throw Error(`Nonopaque crop: ${asset.name}`)
      rgb[3 * i] = rgba[4 * i]
      rgb[3 * i + 1] = rgba[4 * i + 1]
      rgb[3 * i + 2] = rgba[4 * i + 2]
    }
    if (await sha256(rgb) !== asset.rgb_sha256) throw Error(`Decoded RGB mismatch: ${asset.name}`)
    return rgb
  } finally {
    bitmap.close()
  }
}
async function evaluate(model, modelName, caseInfo, nativeCase, target, rgb) {
  let input
  try {
    const flat = modelName === 'sface' ? Float32Array.from(rgb) :
      Float32Array.from(rgb, value => (value - 127.5) / 128)
    const inputHash = await sha256(new Uint8Array(flat.buffer))
    if (inputHash !== nativeCase.input_nhwc_f32_sha256) throw Error(`${modelName} input tensor hash mismatch`)
    input = tf.tensor4d(flat, [1, 112, 112, 3])
    const run = modelName === 'sface' ?
      x => model.execute({ data: x }, 'Identity') : x => model.predict(x)
    const forwardStart = performance.now()
    const rawTensor = run(input)
    if (Array.isArray(rawTensor)) throw Error('Expected one embedding tensor')
    const raw = Array.from(await rawTensor.data())
    rawTensor.dispose()
    const forwardMs = performance.now() - forwardStart
    const gradFn = tf.grad(x => {
      const output = run(x)
      const normalized = tf.div(output, tf.norm(output, 'euclidean', 1, true))
      return tf.sub(1, tf.sum(tf.mul(normalized, target)))
    })
    const gradientStart = performance.now()
    const gradientTensor = gradFn(input)
    const gradient = Array.from(await gradientTensor.data())
    gradientTensor.dispose()
    const gradientMs = performance.now() - gradientStart
    const finiteNonzero = raw.every(Number.isFinite) && gradient.length === flat.length &&
      gradient.every(Number.isFinite) && gradient.some(value => value !== 0)
    let metrics, checks
    if (modelName === 'sface') {
      metrics = {
        rawMaxVsOpenCV: maxDifference(raw, nativeCase.opencv_raw),
        unitCosineVsOpenCV: cosine(unit(raw), unit(nativeCase.opencv_raw)),
        gradientCosineVsOnnx2Torch: cosine(gradient, nativeCase.gradient_nhwc),
      }
      checks = {
        raw: metrics.rawMaxVsOpenCV <= limits.sfaceRawMax,
        unit: metrics.unitCosineVsOpenCV >= limits.sfaceUnitCosineMin,
        gradient: metrics.gradientCosineVsOnnx2Torch >= limits.sfaceGradientCosineMin,
        finiteNonzero,
      }
    } else {
      const normalized = unit(raw)
      metrics = {
        rawMaxVsFloat32Clone: maxDifference(raw, nativeCase.clone_raw),
        unitCosineVsAuthor: cosine(normalized, nativeCase.author_unit),
        unitMaxVsAuthor: maxDifference(normalized, nativeCase.author_unit),
        gradientCosineVsFloat32Clone: cosine(gradient, nativeCase.gradient_nhwc),
      }
      checks = {
        raw: metrics.rawMaxVsFloat32Clone <= limits.ghostRawMax,
        authorCosine: metrics.unitCosineVsAuthor >= limits.ghostAuthorCosineMin,
        authorMax: metrics.unitMaxVsAuthor <= limits.ghostAuthorUnitMax,
        gradient: metrics.gradientCosineVsFloat32Clone >= limits.ghostGradientCosineMin,
        finiteNonzero,
      }
    }
    return { identity: caseInfo.identity, model: modelName, crop_png_sha256: caseInfo.crops[modelName].png_sha256,
      decoded_rgb_sha256: caseInfo.crops[modelName].rgb_sha256, input_sha256: inputHash,
      raw_length: raw.length, gradient_length: gradient.length, metrics, checks,
      pass: Object.values(checks).every(Boolean), timings_ms: { forward: forwardMs, gradient: gradientMs },
      memory_after_case: tf.memory() }
  } finally {
    input?.dispose()
  }
}

onmessage = async () => {
  let sface, ghost, sfaceTarget, ghostTarget
  const memory = {}
  try {
    const [meta, manifest, sfaceRef, ghostRef] = await Promise.all([
      json('/probe-meta.json'), json('/manifest.json'), json('/sface-reference.json'), json('/ghost-reference.json'),
    ])
    if (manifest.schema_version !== 1 || manifest.cases.map(c => c.identity).join(',') !== 'frll-024,frll-036' ||
        sfaceRef.model !== 'sface' || ghostRef.model !== 'ghost' ||
        sfaceRef.cases.length !== 2 || ghostRef.cases.length !== 2) throw Error('Unexpected development fixture schema')
    const backendStart = performance.now()
    if (!await tf.setBackend('webgl')) throw Error('WebGL unavailable in this worker')
    await tf.ready()
    const backendMs = performance.now() - backendStart
    memory.initial = tf.memory()
    progress('Loading SFace GraphModel…')
    const sfaceStart = performance.now()
    sface = await tf.loadGraphModel('/sface/model.json', { streamWeights: true,
      onProgress: fraction => progress(`Loading SFace weights: ${Math.round(100 * fraction)}%…`) })
    const sfaceLoadMs = performance.now() - sfaceStart
    if (sface.inputs[0]?.name !== 'data' || sface.outputs[0]?.name !== 'Identity') throw Error('Wrong SFace API signature')
    memory.after_sface_load = tf.memory()
    progress('Loading GhostFaceNet Layers model…')
    const ghostStart = performance.now()
    ghost = await tf.loadLayersModel('/ghost/model.json')
    const ghostLoadMs = performance.now() - ghostStart
    if (ghost.inputs.length !== 1 || ghost.outputs.length !== 1) throw Error('Wrong Ghost model signature')
    memory.after_both_loaded = tf.memory()
    sfaceTarget = tf.tensor2d(sfaceRef.target_unit, [1, 128])
    ghostTarget = tf.tensor2d(ghostRef.target_unit, [1, 512])
    const cases = []
    for (let index = 0; index < 2; index++) {
      const item = manifest.cases[index]
      if (sfaceRef.cases[index].identity !== item.identity || ghostRef.cases[index].identity !== item.identity)
        throw Error('Reference identity order changed')
      for (const name of ['sface', 'ghost']) {
        progress(`${item.identity}: decoding ${name} crop and computing forward/gradient…`)
        const rgb = await cropRgb(item.crops[name])
        const ref = name === 'sface' ? sfaceRef.cases[index] : ghostRef.cases[index]
        cases.push(await evaluate(name === 'sface' ? sface : ghost, name, item, ref,
          name === 'sface' ? sfaceTarget : ghostTarget, rgb))
      }
    }
    memory.before_disposal = tf.memory()
    sfaceTarget.dispose(); sfaceTarget = undefined
    ghostTarget.dispose(); ghostTarget = undefined
    sface.dispose(); sface = undefined
    ghost.dispose(); ghost = undefined
    await tf.nextFrame()
    memory.after_disposal = tf.memory()
    const report = {
      schema_version: 1, purpose: 'two_model_real_development_crop_component_parity',
      scope: 'Four aligned development crops, two model outputs and input gradients; no optimizer or privacy claim',
      backend: tf.getBackend(), tfjs_version: tf.version.tfjs,
      local_asset_hashes: meta, limits, cases, timings_ms: { backend: backendMs,
        sface_model_load: sfaceLoadMs, ghost_model_load: ghostLoadMs }, memory,
      all_pass: cases.every(c => c.pass) && memory.after_disposal.numTensors === 0 &&
        memory.after_disposal.numBytes === 0,
      memory_note: 'tf.memory excludes total browser/process RAM; WebGL free texture cache may remain allocated.',
    }
    postMessage({ type: 'result', report })
  } catch (error) {
    postMessage({ type: 'error', error: String(error?.stack || error), memory })
  } finally {
    sfaceTarget?.dispose(); ghostTarget?.dispose(); sface?.dispose(); ghost?.dispose()
  }
}
