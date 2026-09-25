/* Local-only synthetic GhostFaceNet TF.js gradient probe. No image files. */
importScripts('/tfjs/tf.min.js')

const limits = Object.freeze({
  authorUnitCosineMin: 0.999,
  authorUnitMaxDifferenceMax: 0.02,
  float32RawMaxDifferenceMax: 0.001,
  float32GradientCosineMin: 0.99,
  finiteDifferenceMinimumPasses: 2,
  finiteDifferenceAbsolute: 1e-5,
  finiteDifferenceRelative: 0.25,
})
const progress = text => postMessage({ type: 'progress', text })
const diff = (a, b) => Math.max(...a.map((value, i) => Math.abs(value - b[i])))
function cosine(a, b) {
  let dot = 0, aa = 0, bb = 0
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; aa += a[i] ** 2; bb += b[i] ** 2 }
  return dot / Math.sqrt(aa * bb)
}
function normed(raw) {
  const size = Math.sqrt(raw.reduce((sum, value) => sum + value * value, 0))
  return raw.map(value => value / size)
}
function probeLoss(model, flat, target) {
  return tf.tidy(() => {
    const input = tf.tensor4d(flat, [1, 112, 112, 3])
    const raw = model.predict(input)
    const unit = tf.div(raw, tf.norm(raw, 'euclidean', 1, true))
    return tf.sub(1, tf.sum(tf.mul(unit, target))).dataSync()[0]
  })
}
onmessage = async ({ data }) => {
  let model, input, target
  try {
    const backend = data.backend
    if (!['cpu', 'webgl'].includes(backend)) throw new Error('Unsupported backend request')
    const [meta, reference] = await Promise.all([
      fetch('/probe-meta.json', { cache: 'no-store' }).then(r => { if (!r.ok) throw Error('Metadata fetch failed'); return r.json() }),
      fetch('/float32-reference.json', { cache: 'no-store' }).then(r => { if (!r.ok) throw Error('Reference fetch failed'); return r.json() }),
    ])
    const backendStart = performance.now()
    if (!await tf.setBackend(backend)) throw new Error(`${backend} backend unavailable in this worker`)
    await tf.ready()
    const backendMs = performance.now() - backendStart
    progress(`Loading ${meta.variant} Layers model on ${tf.getBackend()}…`)
    const loadStart = performance.now()
    model = await tf.loadLayersModel('/model/model.json')
    const loadMs = performance.now() - loadStart
    const flat = new Float32Array(112 * 112 * 3)
    for (let i = 0; i < flat.length; i++) flat[i] = ((i % 256) - 127.5) / 128
    const inputHashBytes = new Uint8Array(await crypto.subtle.digest('SHA-256', flat.buffer))
    const inputHash = Array.from(inputHashBytes, byte => byte.toString(16).padStart(2, '0')).join('')
    if (inputHash !== reference.normalized_tensor_sha256) throw new Error(`Synthetic input hash mismatch: ${inputHash}`)
    input = tf.tensor4d(flat, [1, 112, 112, 3])
    target = tf.tensor2d(reference.target, [1, 512])
    progress('Running forward inference…')
    const forwardStart = performance.now()
    const rawTensor = model.predict(input)
    const raw = Array.from(await rawTensor.data())
    rawTensor.dispose()
    const forwardMs = performance.now() - forwardStart
    if (raw.length !== 512 || !raw.every(Number.isFinite)) throw new Error('Nonfinite or wrong-shape raw output')
    const unit = normed(raw)
    const authorCosine = cosine(unit, reference.author_unit_output)
    const authorMaxDiff = diff(unit, reference.author_unit_output)
    progress('Computing scalar cosine-loss input gradient…')
    const gradientStart = performance.now()
    const gradFn = tf.grad(x => {
      const output = model.predict(x)
      const normalized = tf.div(output, tf.norm(output, 'euclidean', 1, true))
      return tf.sub(1, tf.sum(tf.mul(normalized, target)))
    })
    const gradientTensor = gradFn(input)
    const gradient = Array.from(await gradientTensor.data())
    gradientTensor.dispose()
    const gradientMs = performance.now() - gradientStart
    const finiteGradient = gradient.length === flat.length && gradient.every(Number.isFinite)
    const nonzeroGradient = finiteGradient && gradient.some(value => value !== 0)
    progress('Checking three gradient coordinates by finite differences…')
    const fdStart = performance.now()
    const fdChecks = []
    for (const check of reference.finite_difference.checks.slice(0, 3)) {
      const i = check.index
      const epsilon = reference.finite_difference.epsilon
      const plus = flat.slice(), minus = flat.slice()
      plus[i] += epsilon
      minus[i] -= epsilon
      const estimate = (probeLoss(model, plus, target) - probeLoss(model, minus, target)) / (2 * epsilon)
      const analytic = gradient[i]
      const passed = Number.isFinite(estimate) && Number.isFinite(analytic) &&
        (Math.abs(estimate - analytic) <= limits.finiteDifferenceAbsolute ||
         Math.abs(estimate - analytic) <= limits.finiteDifferenceRelative * Math.max(Math.abs(estimate), Math.abs(analytic)))
      fdChecks.push({ index: i, estimate, analytic, passed })
    }
    const fdMs = performance.now() - fdStart
    const float32 = meta.variant === 'float32'
    const rawMaxDifference = float32 ? diff(raw, reference.float32_raw_output) : null
    const gradientCosine = float32 && finiteGradient && nonzeroGradient
      ? cosine(gradient, reference.float32_input_gradient) : null
    const checks = {
      author_unit: authorCosine >= limits.authorUnitCosineMin && authorMaxDiff <= limits.authorUnitMaxDifferenceMax,
      finite_nonzero_gradient: finiteGradient && nonzeroGradient,
      finite_difference: fdChecks.filter(check => check.passed).length >= limits.finiteDifferenceMinimumPasses,
      float32_raw: float32 ? rawMaxDifference <= limits.float32RawMaxDifferenceMax : null,
      float32_gradient: float32 ? gradientCosine >= limits.float32GradientCosineMin : null,
    }
    const report = {
      schema_version: 1, variant: meta.variant, backend_requested: backend, backend_actual: tf.getBackend(),
      tfjs_version: tf.version.tfjs, input_sha256: inputHash,
      model_sha256: meta.model_sha256, reference_sha256: meta.reference_sha256,
      limits, checks, overall_pass: Object.values(checks).every(value => value !== false),
      author_unit_cosine: authorCosine, author_unit_max_difference: authorMaxDiff,
      float32_raw_max_difference: rawMaxDifference, float32_gradient_cosine: gradientCosine,
      gradient_l2: Math.sqrt(gradient.reduce((sum, value) => sum + value * value, 0)),
      finite_difference: fdChecks,
      timings_ms: { backend: backendMs, model_load: loadMs, forward: forwardMs, gradient: gradientMs, finite_difference: fdMs },
      memory: tf.memory(),
    }
    postMessage({ type: 'result', report })
  } catch (error) {
    postMessage({ type: 'error', error: String(error?.stack || error) })
  } finally {
    input?.dispose(); target?.dispose(); model?.dispose()
  }
}
