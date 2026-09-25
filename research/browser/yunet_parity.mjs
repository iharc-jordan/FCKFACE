const status = document.getElementById('status')
const result = document.getElementById('result')
let worker
function stop() {
  worker?.terminate()
  worker = undefined
  status.textContent = 'Cancelled. Worker stopped.'
}
function start() {
  stop()
  status.textContent = 'Loading local test assets…'
  result.textContent = ''
  window.__YUNET_PARITY_RESULT__ = undefined
  window.__YUNET_PARITY_ERROR__ = undefined
  worker = new Worker('/yunet_worker.mjs', { type: 'module' })
  worker.onmessage = event => {
    if (event.data.type === 'progress') status.textContent = event.data.text
    if (event.data.type === 'result') {
      window.__YUNET_PARITY_RESULT__ = event.data.report
      result.textContent = JSON.stringify(event.data.report, null, 2)
      status.textContent = 'Parity probe complete.'
      worker = undefined
    }
    if (event.data.type === 'error') {
      window.__YUNET_PARITY_ERROR__ = event.data.error
      result.textContent = event.data.error
      status.textContent = 'Parity probe failed.'
      worker = undefined
    }
  }
  worker.onerror = error => {
    window.__YUNET_PARITY_ERROR__ = error.message
    result.textContent = error.message
    status.textContent = 'Worker failed.'
    worker = undefined
  }
}
document.getElementById('start').addEventListener('click', start)
document.getElementById('cancel').addEventListener('click', stop)
start()
