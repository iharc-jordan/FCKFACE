/// <reference lib="webworker" />
import { MAX_PIXELS, MAX_BYTES } from './image-utils'
import { isValidBox, METHOD_VERSION, type ProcessInput, type ProcessMessage } from './processing-contract'

const report = (message: ProcessMessage) => self.postMessage(message)

self.onmessage = async (event: MessageEvent<ProcessInput>) => {
  let bitmap: ImageBitmap | undefined
  try {
    const { image, selectedBox, methodVersion } = event.data
    if (methodVersion !== METHOD_VERSION || !isValidBox(selectedBox)) throw new Error('Select one face region before processing.')
    if (image.size > MAX_BYTES) throw new Error('The image exceeds the 20 MB limit.')
    report({ type: 'progress', percent: 12, label: 'Reading image' })
    bitmap = await createImageBitmap(image, { imageOrientation: 'from-image' })
    const { width, height } = bitmap
    if (!width || !height || width * height > MAX_PIXELS) throw new Error('The image exceeds the 24 megapixel limit.')
    if (typeof OffscreenCanvas === 'undefined') throw new Error('This browser does not support local canvas processing in a worker.')
    const canvas = new OffscreenCanvas(width, height)
    const context = canvas.getContext('2d')
    if (!context) throw new Error('Could not prepare the image canvas.')
    context.drawImage(bitmap, 0, 0)
    bitmap.close()
    bitmap = undefined
    report({ type: 'progress', percent: 45, label: 'Applying graphic dot preview' })

    const { x, y, width: boxWidth, height: boxHeight } = selectedBox
    const left = x * width
    const top = y * height
    const faceWidth = boxWidth * width
    const faceHeight = boxHeight * height
    const spacing = Math.max(8, Math.min(22, faceWidth / 34))
    const radius = spacing * 0.24

    context.save()
    context.beginPath()
    context.ellipse(left + faceWidth / 2, top + faceHeight / 2, faceWidth * 0.48, faceHeight * 0.48, 0, 0, Math.PI * 2)
    context.clip()
    context.fillStyle = '#a9db2e'
    context.globalAlpha = 0.67
    for (let row = 0, py = top + spacing / 2; py < top + faceHeight; row++, py += spacing) {
      for (let px = left + spacing / 2 + (row % 2) * spacing / 2; px < left + faceWidth; px += spacing) {
        context.beginPath()
        context.arc(px, py, radius, 0, Math.PI * 2)
        context.fill()
      }
    }
    context.restore()
    report({ type: 'progress', percent: 78, label: 'Exporting JPEG' })
    // Canvas export creates a new JPEG from pixels, without source EXIF or other identifying metadata.
    const output = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.92 })
    if (output.type !== 'image/jpeg' || !output.size) throw new Error('Could not export a JPEG.')
    report({ type: 'result', output, width, height, warnings: [
      'Graphic dot effect only. Facial recognition resistance has not been validated.',
      'JPEG output has no source metadata and does not preserve PNG/WebP transparency.',
    ] })
  } catch (error) {
    report({ type: 'error', message: error instanceof Error ? error.message : 'Processing failed.' })
  } finally {
    bitmap?.close()
  }
}
