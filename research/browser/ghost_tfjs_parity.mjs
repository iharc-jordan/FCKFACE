const status = document.getElementById('status')
const result = document.getElementById('result')
let worker
function cancel() {
  worker?.terminate()
  worker = undefined
  status.textContent = 'Worker stopped. Choose a backend to restart.'
}
function start(backend, benchmark = false) {
  cancel()
  result.textContent = ''
  status.textContent = `Starting isolated ${backend} worker…`
  window.__GHOST_TFJS_RESULT__ = undefined
  window.__GHOST_TFJS_ERROR__ = undefined
  window.__GHOST_TFJS_BENCHMARK_RESULT__ = undefined
  worker = new Worker(benchmark ? '/ghost_tfjs_warm_worker.js' : '/ghost_tfjs_worker.js')
  worker.onmessage = ({ data }) => {
    if (data.type === 'progress') status.textContent = data.text
    if (data.type === 'result') {
      if (benchmark) window.__GHOST_TFJS_BENCHMARK_RESULT__ = data.report
      else window.__GHOST_TFJS_RESULT__ = data.report
      result.textContent = JSON.stringify(data.report, null, 2)
      const passed = benchmark ? data.report.all_numerically_valid : data.report.overall_pass
      status.textContent = passed === true ? 'Diagnostic pass.' : 'Diagnostic failed or incomplete.'
      worker?.terminate()
      worker = undefined
    }
    if (data.type === 'error') {
      window.__GHOST_TFJS_ERROR__ = data.error
      result.textContent = data.error
      status.textContent = 'Probe failed.'
      worker?.terminate()
      worker = undefined
    }
  }
  worker.onerror = error => {
    window.__GHOST_TFJS_ERROR__ = error.message
    result.textContent = error.message
    status.textContent = 'Worker error.'
    worker?.terminate()
    worker = undefined
  }
  worker.postMessage({ backend })
}
document.getElementById('cpu').addEventListener('click', () => start('cpu'))
document.getElementById('webgl').addEventListener('click', () => start('webgl'))
document.getElementById('warm').addEventListener('click', () => start('webgl', true))
document.getElementById('cancel').addEventListener('click', cancel)
