/* Synthetic-only SFace TF.js GraphModel probe. Never fetches photographs. */
importScripts('/tfjs/tf.min.js')

const limits = Object.freeze({ rawMax: 1e-3, unitCosineMin: .99999,
  gradientCosineMin: .99, fdAbs: 1e-5, fdRelative: .25, fdPassMin: 2 })
const progress = text => postMessage({ type: 'progress', text })

function maxDifference(a, b) {
  let result = 0
  for (let i = 0; i < a.length; i++) result = Math.max(result, Math.abs(a[i] - b[i]))
  return result
}
function cosine(a, b) {
  let dot = 0, aa = 0, bb = 0
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; aa += a[i] ** 2; bb += b[i] ** 2 }
  return dot / Math.sqrt(aa * bb)
}
function unit(a) {
  let size = 0
  for (const value of a) size += value * value
  size = Math.sqrt(size)
  return a.map(value => value / size)
}
function seedZeroRgb() {
  // NumPy PCG64 seed 0 is frozen in the reference; the bytes are served as a
  // synthetic binary asset so JS never substitutes a different RNG.
  return fetch('/synthetic-rgb.bin', { cache: 'no-store' }).then(async response => {
    if (!response.ok) throw Error('Synthetic tensor fetch failed')
    return new Uint8Array(await response.arrayBuffer())
  })
}
async function sha256(bytes) {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))
  return Array.from(digest, byte => byte.toString(16).padStart(2, '0')).join('')
}

onmessage = async ({ data }) => {
  let model, input, target
  const memory = {}
  try {
    const backend = data.backend
    const mode = data.mode || 'parity'
    if (!['cpu', 'webgl'].includes(backend) || !['parity', 'warm'].includes(mode)) throw Error('Invalid probe request')
    const [meta, reference, rgb] = await Promise.all([
      fetch('/probe-meta.json', { cache: 'no-store' }).then(r => r.json()),
      fetch('/cosine-reference.json', { cache: 'no-store' }).then(r => r.json()),
      seedZeroRgb(),
    ])
    if (reference.schema_version !== 1 || rgb.length !== 112 * 112 * 3 ||
        await sha256(rgb) !== reference.rgb_u8_sha256) throw Error('Frozen synthetic RGB mismatch')
    const flat = Float32Array.from(rgb)
    const inputHash = await sha256(new Uint8Array(flat.buffer))
    if (inputHash !== reference.nhwc_f32_sha256) throw Error('Frozen NHWC float32 mismatch')
    const backendStart = performance.now()
    if (!await tf.setBackend(backend)) throw Error(`${backend} backend unavailable`)
    await tf.ready()
    const backendMs = performance.now() - backendStart
    memory.initial = tf.memory()
    progress(`Loading SFace GraphModel on ${tf.getBackend()}…`)
    const loadStart = performance.now()
    model = await tf.loadGraphModel('/model/model.json', {
      streamWeights: true,
      onProgress: fraction => progress(`Loading model weights: ${Math.round(100 * fraction)}%…`),
    })
    const loadMs = performance.now() - loadStart
    // The GraphModel API strips the ":0" suffix from model.json signature names.
    if (model.inputs.length !== 1 || model.outputs.length !== 1 ||
        model.inputs[0].name !== 'data' || model.outputs[0].name !== 'Identity') {
      throw Error(`Unexpected GraphModel signature: ${JSON.stringify({ inputs: model.inputs, outputs: model.outputs })}`)
    }
    memory.loaded = tf.memory()
    input = tf.tensor4d(flat, [1, 112, 112, 3]) // RGB 0..255; ONNX graph normalizes internally.
    target = tf.tensor1d(reference.target_unit)
    const rawFor = x => {
      const result = model.execute({ data: x }, 'Identity')
      if (Array.isArray(result) || result.shape.length !== 2 || result.shape[0] !== 1 || result.shape[1] !== 128) {
        result?.dispose?.()
        throw Error('Expected one [1,128] GraphModel output tensor')
      }
      return result
    }
    const objective = x => {
      const raw = rawFor(x)
      const normalized = tf.div(raw, tf.norm(raw))
      return tf.sub(1, tf.sum(tf.mul(normalized, target)))
    }
    const gradFn = tf.grad(objective)
    const forward = async () => {
      const started = performance.now()
      const result = rawFor(input)
      const values = Array.from(await result.data())
      result.dispose()
      return { ms: performance.now() - started, values,
        rawMax: maxDifference(values, reference.raw_opencv),
        unitCosine: cosine(unit(values), unit(reference.raw_opencv)) }
    }
    const gradient = async () => {
      const started = performance.now()
      const result = gradFn(input)
      const values = Array.from(await result.data())
      result.dispose()
      return { ms: performance.now() - started, values,
        cosine: cosine(values, reference.gradient_nhwc),
        finiteNonzero: values.length === flat.length && values.every(Number.isFinite) && values.some(v => v !== 0) }
    }
    const firstForward = await forward()
    if (firstForward.values.length !== 128 || !firstForward.values.every(Number.isFinite)) throw Error('Invalid raw output')
    progress('Computing SFace cosine-loss input gradient…')
    const firstGradient = await gradient()
    const checks = {
      rawMax: firstForward.rawMax <= limits.rawMax,
      unitCosine: firstForward.unitCosine >= limits.unitCosineMin,
      finiteNonzeroGradient: firstGradient.finiteNonzero,
      gradientCosine: firstGradient.cosine >= limits.gradientCosineMin,
    }
    const fd = []
    if (mode === 'parity') {
      progress('Checking three synthetic gradient coordinates…')
      for (const check of reference.finite_difference_coordinates) {
        const index = check.index_nhwc, step = reference.finite_difference_step
        const plus = flat.slice(), minus = flat.slice()
        plus[index] += step; minus[index] -= step
        const valueFor = data => tf.tidy(() => {
          const x = tf.tensor4d(data, [1, 112, 112, 3])
          return objective(x).dataSync()[0]
        })
        const estimate = (valueFor(plus) - valueFor(minus)) / (2 * step)
        const analytic = firstGradient.values[index]
        const passed = Number.isFinite(estimate) && Number.isFinite(analytic) &&
          (Math.abs(estimate - analytic) <= limits.fdAbs ||
           Math.abs(estimate - analytic) <= limits.fdRelative * Math.max(Math.abs(estimate), Math.abs(analytic)))
        fd.push({ index_nhwc: index, estimate, analytic, passed })
      }
      checks.finiteDifference = fd.filter(row => row.passed).length >= limits.fdPassMin
    }
    const runs = []
    if (mode === 'warm' && Object.values(checks).every(Boolean)) {
      for (let i = 0; i < 3; i++) {
        progress(`Warm forward and gradient ${i + 1}/3…`)
        const next = { forward: await forward(), gradient: await gradient() }
        runs.push({ forwardMs: next.forward.ms, gradientMs: next.gradient.ms,
          rawMax: next.forward.rawMax, unitCosine: next.forward.unitCosine,
          gradientCosine: next.gradient.cosine, finiteNonzero: next.gradient.finiteNonzero })
      }
    }
    memory.beforeDisposal = tf.memory()
    input.dispose(); input = undefined
    target.dispose(); target = undefined
    model.dispose(); model = undefined
    await tf.nextFrame()
    memory.afterDisposal = tf.memory()
    const report = {
      schema_version: 1, purpose: mode === 'warm' ? 'sface_tfjs_warm_benchmark' : 'sface_tfjs_synthetic_parity',
      backend_requested: backend, backend_actual: tf.getBackend(), tfjs_version: tf.version.tfjs,
      model_sha256: meta.model_sha256, reference_sha256: meta.reference_sha256,
      weight_shards_sha256: meta.weight_shards_sha256, tfjs_sha256: meta.tfjs_sha256,
      input_sha256: inputHash, limits, checks, overall_pass: Object.values(checks).every(Boolean) &&
        runs.every(run => run.rawMax <= limits.rawMax && run.unitCosine >= limits.unitCosineMin &&
          run.gradientCosine >= limits.gradientCosineMin && run.finiteNonzero),
      raw_max_difference: firstForward.rawMax, unit_cosine: firstForward.unitCosine,
      gradient_cosine: firstGradient.cosine,
      gradient_l2: Math.sqrt(firstGradient.values.reduce((sum, value) => sum + value * value, 0)),
      finite_difference: fd,
      timings_ms: { backend: backendMs, model_load: loadMs,
        forward: firstForward.ms, gradient: firstGradient.ms },
      warm_runs: runs, memory,
      memory_note: 'tf.memory excludes browser process memory; GPU texture cache may persist after disposal.',
    }
    postMessage({ type: 'result', report })
  } catch (error) {
    postMessage({ type: 'error', error: String(error?.stack || error), memory })
  } finally {
    input?.dispose(); target?.dispose(); model?.dispose()
  }
}
