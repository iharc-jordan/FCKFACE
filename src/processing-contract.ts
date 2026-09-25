/** Coordinates are fractions of the visually oriented image, in [0, 1]. */
export type FaceBox = { x: number; y: number; width: number; height: number }

export const METHOD_VERSION = 'graphic-dots-preview-0.1' as const

export type ProcessInput = {
  image: Blob
  selectedBox: FaceBox
  methodVersion: typeof METHOD_VERSION
}

export type ProcessMessage =
  | { type: 'progress'; percent: number; label: string }
  | { type: 'result'; output: Blob; width: number; height: number; warnings: string[] }
  | { type: 'error'; message: string }

export function isValidBox(box: FaceBox): boolean {
  return [box.x, box.y, box.width, box.height].every(Number.isFinite) &&
    box.x >= 0 && box.y >= 0 && box.width > 0 && box.height > 0 &&
    box.x + box.width <= 1.000001 && box.y + box.height <= 1.000001
}
