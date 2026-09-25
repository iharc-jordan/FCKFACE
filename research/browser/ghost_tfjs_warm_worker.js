/* Bounded WebGL warm timing on the same synthetic RGB112 tensor; no photos. */
importScripts('/tfjs/tf.min.js')

const progress = text => postMessage({ type: 'progress', text })
const maxAbs = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])))
function cosine(a, b) {
  let dot = 0, aa = 0, bb = 0
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; aa += a[i] ** 2; bb += b[i] ** 2 }
  return dot / Math.sqrt(aa * bb)
}

onmessage = async () => {
  let model, input, target
  const memory = {}
  try {
    const [meta, reference] = await Promise.all([
      fetch('/probe-meta.json').then(r => r.json()),
      fetch('/float32-reference.json').then(r => r.json()),
    ])
    if (meta.variant !== 'float32') throw new Error('Warm probe requires the frozen float32 model')
    if (!await tf.setBackend('webgl')) throw new Error('WebGL backend unavailable in worker')
    await tf.ready()
    memory.initial = tf.memory()
    progress('Loading frozen float32 Layers model on WebGL…')
    model = await tf.loadLayersModel('/model/model.json')
    const flat = new Float32Array(112 * 112 * 3)
    for (let i = 0; i < flat.length; i++) flat[i] = ((i % 256) - 127.5) / 128
    const inputHashBytes = new Uint8Array(await crypto.subtle.digest('SHA-256', flat.buffer))
    const inputHash = Array.from(inputHashBytes, byte => byte.toString(16).padStart(2, '0')).join('')
    if (inputHash !== reference.normalized_tensor_sha256) throw new Error('Synthetic input hash mismatch')
    input = tf.tensor4d(flat, [1, 112, 112, 3])
    target = tf.tensor2d(reference.target, [1, 512])
    memory.loaded = tf.memory()
    const gradFn = tf.grad(x => {
      const output = model.predict(x)
      const unit = tf.div(output, tf.norm(output, 'euclidean', 1, true))
      return tf.sub(1, tf.sum(tf.mul(unit, target)))
    })
    async function forward() {
      const start = performance.now()
      const tensor = model.predict(input)
      const values = Array.from(await tensor.data())
      tensor.dispose()
      return { ms: performance.now() - start, raw_max_difference: maxAbs(values, reference.float32_raw_output) }
    }
    async function gradient() {
      const start = performance.now()
      const tensor = gradFn(input)
      const values = Array.from(await tensor.data())
      tensor.dispose()
      return { ms: performance.now() - start,
        cosine: cosine(values, reference.float32_input_gradient),
        finite_nonzero: values.every(Number.isFinite) && values.some(value => value !== 0) }
    }
    progress('Warming forward and gradient kernels once…')
    const warmup = { forward: await forward(), gradient: await gradient() }
    memory.after_warmup = tf.memory()
    const runs = []
    for (let i = 0; i < 3; i++) {
      progress(`Timed warm run ${i + 1}/3…`)
      runs.push({ forward: await forward(), gradient: await gradient() })
    }
    memory.before_disposal = tf.memory()
    const valid = [warmup, ...runs].every(run =>
      run.forward.raw_max_difference <= 0.001 && run.gradient.cosine >= 0.99 && run.gradient.finite_nonzero)
    input.dispose(); input = undefined
    target.dispose(); target = undefined
    model.dispose(); model = undefined
    await tf.nextFrame()
    memory.after_disposal = tf.memory()
    const report = {
      schema_version: 1, purpose: 'three_warm_webgl_forward_gradient_timings',
      backend: tf.getBackend(), tfjs_version: tf.version.tfjs,
      model_sha256: meta.model_sha256, reference_sha256: meta.reference_sha256,
      input_sha256: inputHash, numerical_limits_unchanged: { raw_max: 0.001, gradient_cosine_min: 0.99 },
      all_numerically_valid: valid, warmup, runs, memory,
      note: 'tf.memory excludes browser process memory; GPU texture cache may survive model disposal.',
    }
    postMessage({ type: 'result', report })
  } catch (error) {
    postMessage({ type: 'error', error: String(error?.stack || error), memory })
  } finally {
    input?.dispose(); target?.dispose(); model?.dispose()
  }
}
