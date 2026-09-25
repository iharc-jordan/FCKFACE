const status = document.getElementById('status')
const result = document.getElementById('result')
let worker
function cancel() {
  worker?.terminate()
  worker = undefined
  status.textContent = 'Worker stopped. Choose a backend to restart.'
}
function start(backend, mode = 'parity') {
  cancel()
  result.textContent = ''
  status.textContent = `Starting isolated ${backend} worker…`
  window.__SFACE_TFJS_RESULT__ = undefined
  window.__SFACE_TFJS_WARM_RESULT__ = undefined
  window.__SFACE_TFJS_ERROR__ = undefined
  worker = new Worker('/sface_tfjs_worker.js')
  worker.onmessage = ({ data }) => {
    if (data.type === 'progress') status.textContent = data.text
    if (data.type === 'result') {
      if (mode === 'warm') window.__SFACE_TFJS_WARM_RESULT__ = data.report
      else window.__SFACE_TFJS_RESULT__ = data.report
      result.textContent = JSON.stringify(data.report, null, 2)
      status.textContent = data.report.overall_pass ? 'Diagnostic pass.' : 'Diagnostic failed or incomplete.'
      worker?.terminate()
      worker = undefined
    }
    if (data.type === 'error') {
      window.__SFACE_TFJS_ERROR__ = data.error
      result.textContent = data.error
      status.textContent = 'Probe failed.'
      worker?.terminate()
      worker = undefined
    }
  }
  worker.onerror = error => {
    window.__SFACE_TFJS_ERROR__ = error.message
    result.textContent = error.message
    status.textContent = 'Worker error.'
    worker?.terminate()
    worker = undefined
  }
  worker.postMessage({ backend, mode })
}
document.getElementById('cpu').addEventListener('click', () => start('cpu'))
document.getElementById('webgl').addEventListener('click', () => start('webgl'))
document.getElementById('warm').addEventListener('click', () => start('webgl', 'warm'))
document.getElementById('cancel').addEventListener('click', cancel)
