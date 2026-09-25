const status = document.getElementById('status')
const result = document.getElementById('result')
let worker
function cancel() {
  worker?.terminate()
  worker = undefined
  status.textContent = 'Worker stopped. Ready to restart.'
}
function start() {
  cancel()
  result.textContent = ''
  status.textContent = 'Starting isolated WebGL worker…'
  window.__COMBINED_CROP_RESULT__ = undefined
  window.__COMBINED_CROP_ERROR__ = undefined
  worker = new Worker('/combined_crop_worker.js')
  worker.onmessage = ({ data }) => {
    if (data.type === 'progress') status.textContent = data.text
    if (data.type === 'result') {
      window.__COMBINED_CROP_RESULT__ = data.report
      result.textContent = JSON.stringify(data.report, null, 2)
      status.textContent = data.report.all_pass ? 'Development crop parity passed.' : 'Development crop parity failed.'
      worker?.terminate()
      worker = undefined
    }
    if (data.type === 'error') {
      window.__COMBINED_CROP_ERROR__ = data.error
      result.textContent = data.error
      status.textContent = 'Probe failed.'
      worker?.terminate()
      worker = undefined
    }
  }
  worker.onerror = error => {
    window.__COMBINED_CROP_ERROR__ = error.message
    result.textContent = error.message
    status.textContent = 'Worker error.'
    worker?.terminate()
    worker = undefined
  }
  worker.postMessage({ run: true })
}
document.getElementById('run').addEventListener('click', start)
document.getElementById('cancel').addEventListener('click', cancel)
