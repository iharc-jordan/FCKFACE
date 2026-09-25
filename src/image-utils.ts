export const MAX_BYTES = 20 * 1024 * 1024
export const MAX_PIXELS = 24_000_000
const ALLOWED = new Set(['image/jpeg', 'image/png', 'image/webp'])

export function validateFile(file: File): string | null {
  if (!ALLOWED.has(file.type)) return 'Choose a JPEG, PNG or WebP image.'
  if (file.size > MAX_BYTES) return 'This file is larger than 20 MB. Choose a smaller image.'
  if (file.size === 0) return 'This file is empty.'
  return null
}

export async function decodeImage(blob: Blob): Promise<ImageBitmap> {
  return createImageBitmap(blob, { imageOrientation: 'from-image' })
}

export function validateDimensions(width: number, height: number): string | null {
  if (width < 1 || height < 1 || width * height > MAX_PIXELS) {
    return 'This image is too large to process here. The limit is 24 megapixels.'
  }
  return null
}
