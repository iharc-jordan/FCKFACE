import { useEffect, useRef, useState, type CSSProperties, type PointerEvent } from 'react'
import { decodeImage, validateDimensions, validateFile } from './image-utils'
import { isValidBox, METHOD_VERSION, type FaceBox, type ProcessMessage } from './processing-contract'

type Photo = { file: File; url: string; width: number; height: number }
type Result = { url: string; width: number; height: number; warnings: string[] }
type Progress = { percent: number; label: string }

const INITIAL_BOX: FaceBox = { x: 0.3, y: 0.16, width: 0.4, height: 0.68 }
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value))

function cropStyle(box: FaceBox): CSSProperties {
  return {
    width: `${100 / box.width}%`,
    height: `${100 / box.height}%`,
    left: `${-100 * box.x / box.width}%`,
    top: `${-100 * box.y / box.height}%`,
  }
}

export function ResearchApp() {
  const [photo, setPhoto] = useState<Photo | null>(null)
  const [box, setBox] = useState<FaceBox | null>(null)
  const [result, setResult] = useState<Result | null>(null)
  const [progress, setProgress] = useState<Progress | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const workerRef = useRef<Worker | null>(null)
  const photoUrlRef = useRef<string | null>(null)
  const resultUrlRef = useRef<string | null>(null)
  const loadIdRef = useRef(0)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const dragRef = useRef<{ x: number; y: number } | null>(null)

  function disposeWorker() {
    workerRef.current?.terminate()
    workerRef.current = null
  }

  function clearResult() {
    if (resultUrlRef.current) URL.revokeObjectURL(resultUrlRef.current)
    resultUrlRef.current = null
    setResult(null)
  }

  function cancel() {
    disposeWorker()
    setProgress(null)
  }

  useEffect(() => () => {
    loadIdRef.current++
    workerRef.current?.terminate()
    if (photoUrlRef.current) URL.revokeObjectURL(photoUrlRef.current)
    if (resultUrlRef.current) URL.revokeObjectURL(resultUrlRef.current)
  }, [])

  async function chooseFile(file: File | undefined) {
    if (!file) return
    const id = ++loadIdRef.current
    cancel()
    clearResult()
    setBox(null)
    setError(null)
    setLoading(false)
    if (photoUrlRef.current) URL.revokeObjectURL(photoUrlRef.current)
    photoUrlRef.current = null
    setPhoto(null)
    const fileError = validateFile(file)
    if (fileError) { setError(fileError); return }
    setLoading(true)
    let bitmap: ImageBitmap | undefined
    try {
      bitmap = await decodeImage(file)
      if (id !== loadIdRef.current) return
      const dimensionError = validateDimensions(bitmap.width, bitmap.height)
      if (dimensionError) { setError(dimensionError); return }
      const url = URL.createObjectURL(file)
      photoUrlRef.current = url
      setPhoto({ file, url, width: bitmap.width, height: bitmap.height })
    } catch {
      if (id === loadIdRef.current) setError('This image could not be opened. Choose another JPEG, PNG or WebP file.')
    } finally {
      bitmap?.close()
      if (id === loadIdRef.current) setLoading(false)
    }
  }

  function pointFromEvent(event: PointerEvent<HTMLDivElement>) {
    const bounds = imageRef.current?.getBoundingClientRect()
    if (!bounds) return null
    return {
      x: clamp((event.clientX - bounds.left) / bounds.width, 0, 1),
      y: clamp((event.clientY - bounds.top) / bounds.height, 0, 1),
    }
  }

  function beginDrag(event: PointerEvent<HTMLDivElement>) {
    if (progress || !photo) return
    const point = pointFromEvent(event)
    if (!point) return
    event.currentTarget.setPointerCapture(event.pointerId)
    dragRef.current = point
    clearResult()
    setBox(null)
    setError(null)
  }

  function continueDrag(event: PointerEvent<HTMLDivElement>) {
    const start = dragRef.current
    const end = pointFromEvent(event)
    if (!start || !end) return
    setBox({ x: Math.min(start.x, end.x), y: Math.min(start.y, end.y), width: Math.abs(start.x - end.x), height: Math.abs(start.y - end.y) })
  }

  function endDrag(event: PointerEvent<HTMLDivElement>) {
    if (!dragRef.current) return
    continueDrag(event)
    dragRef.current = null
    setBox(current => current && current.width >= 0.025 && current.height >= 0.025 ? current : null)
  }

  function updateBox(field: keyof FaceBox, value: number) {
    if (!Number.isFinite(value)) return
    clearResult()
    setBox(current => {
      const next = { ...(current ?? INITIAL_BOX) }
      const fraction = clamp(value / 100, 0, 1)
      if (field === 'x') next.x = clamp(fraction, 0, 1 - next.width)
      else if (field === 'y') next.y = clamp(fraction, 0, 1 - next.height)
      else if (field === 'width') next.width = clamp(fraction, 0.025, 1 - next.x)
      else next.height = clamp(fraction, 0.025, 1 - next.y)
      return next
    })
  }

  function processPhoto() {
    if (!photo || !box || !isValidBox(box)) return
    cancel()
    clearResult()
    setError(null)
    try {
      const worker = new Worker(new URL('./process.worker.ts', import.meta.url), { type: 'module' })
      workerRef.current = worker
      setProgress({ percent: 0, label: 'Starting local processing' })
      worker.onmessage = (event: MessageEvent<ProcessMessage>) => {
        if (worker !== workerRef.current) return
        const message = event.data
        if (message.type === 'progress') setProgress({ percent: message.percent, label: message.label })
        if (message.type === 'error') { setError(message.message); cancel() }
        if (message.type === 'result') {
          const url = URL.createObjectURL(message.output)
          resultUrlRef.current = url
          setResult({ url, width: message.width, height: message.height, warnings: message.warnings })
          cancel()
        }
      }
      worker.onerror = () => { setError('The local processor stopped unexpectedly. Try a smaller image.'); cancel() }
      worker.postMessage({ image: photo.file, selectedBox: box, methodVersion: METHOD_VERSION })
    } catch {
      setError('Could not start local processing in this browser.')
      cancel()
    }
  }

  const disabled = progress !== null
  return (
    <main className="page research-page">
      <div className="eyebrow"><span className="signal" /> LOCAL / UNVALIDATED METHOD PREVIEW</div>
      <div className="intro-grid">
        <div><h1>Image effects under examination.</h1><p className="lede">Choose one photo, mark one face, and inspect a visibly artificial dot effect. This local preview is for visual research. It has not been shown to resist facial recognition.</p></div>
        <div className="privacy-note"><span className="overline">ON THIS DEVICE</span><p>Your photo is processed in a browser worker. No upload, account, analytics or saved photo history.</p></div>
      </div>

      <section className="workflow" aria-labelledby="step-one">
        <div className="section-heading"><span className="step">01</span><div><h2 id="step-one">Choose an image</h2><p>JPEG, PNG or WebP · up to 20 MB and 24 megapixels</p></div></div>
        <label className="file-control"><input type="file" accept="image/jpeg,image/png,image/webp" onChange={event => { void chooseFile(event.target.files?.[0]); event.target.value = '' }} /><span>{photo ? 'Choose another photo' : 'Choose a photo'}</span><span aria-hidden="true">↗</span></label>
        {loading && <p role="status">Reading image…</p>}
        {error && <p className="error" role="alert">{error}</p>}
      </section>

      {photo && <>
        <section className="workflow" aria-labelledby="step-two">
          <div className="section-heading"><span className="step">02</span><div><h2 id="step-two">Mark one face</h2><p>Drag a rectangle over one face. You can also set the region with the keyboard controls below.</p></div></div>
          <div className="selection-layout">
            <div className={`selection-surface${disabled ? ' disabled' : ''}`} onPointerDown={beginDrag} onPointerMove={continueDrag} onPointerUp={endDrag} onPointerCancel={endDrag}>
              <img ref={imageRef} src={photo.url} alt="Selected source photo; drag to mark one face" draggable={false} />
              {box && <div className="face-box" style={{ left: `${box.x * 100}%`, top: `${box.y * 100}%`, width: `${box.width * 100}%`, height: `${box.height * 100}%` }} aria-hidden="true"><span>SELECTED FACE</span></div>}
            </div>
            <div className="selection-tools"><span className="overline">FACE REGION / MANUAL</span><p>Automatic face detection is not part of this preview. Place the rectangle around the face you want to alter.</p>
              <button type="button" className="text-button" disabled={disabled} onClick={() => { clearResult(); setBox(INITIAL_BOX) }}>Start with centered region</button>
              {box && <div className="box-fields">{([['x', 'Left'], ['y', 'Top'], ['width', 'Width'], ['height', 'Height']] as const).map(([field, label]) => <label key={field}>{label}<span><input type="number" min="0" max="100" step="1" value={Math.round(box[field] * 100)} disabled={disabled} onChange={event => updateBox(field, Number(event.target.value))} />%</span></label>)}</div>}
              <p className="detail">{photo.width} × {photo.height} px · Original retained locally</p>
            </div>
          </div>
        </section>

        <section className="workflow" aria-labelledby="step-three">
          <div className="section-heading"><span className="step">03</span><div><h2 id="step-three">Inspect the preview</h2><p>Graphic dots are an appearance study, not a privacy guarantee.</p></div></div>
          <div className="action-row"><button type="button" className="primary-button" onClick={processPhoto} disabled={!box || disabled}>Apply dot preview <span aria-hidden="true">→</span></button>{disabled && <button type="button" className="text-button" onClick={cancel}>Cancel processing</button>}</div>
          {progress && <div className="progress" role="status" aria-live="polite"><span>{progress.label}</span><span>{progress.percent}%</span><progress max="100" value={progress.percent} aria-label="Processing progress" /></div>}
          {result && box && <div className="results">
            <div className="result-heading"><h3>Compare the images</h3><p>Same dimensions: {result.width} × {result.height} px</p></div>
            <div className="comparison"><figure><figcaption>ORIGINAL</figcaption><img src={photo.url} alt="Full original photo" /><div className="crop" style={{ aspectRatio: `${box.width * photo.width} / ${box.height * photo.height}` }}><img src={photo.url} alt="Original selected face crop" style={cropStyle(box)} /></div><span className="crop-caption">SELECTED REGION</span></figure><figure><figcaption>GRAPHIC PREVIEW</figcaption><img src={result.url} alt="Full photo with artificial dots over the selected face" /><div className="crop" style={{ aspectRatio: `${box.width * photo.width} / ${box.height * photo.height}` }}><img src={result.url} alt="Selected face crop with artificial dots" style={cropStyle(box)} /></div><span className="crop-caption">SELECTED REGION</span></figure></div>
            <a className="download-button" href={result.url} download="fckface-research.jpg">Download JPEG <span aria-hidden="true">↓</span></a>
            <p className="detail">{result.warnings.join(' ')}</p>
          </div>}
        </section>
      </>}
      <footer className="page-footer"><span>FCKFACE / Local research preview</span><span>No recognition claim · No external transfer</span></footer>
    </main>
  )
}
