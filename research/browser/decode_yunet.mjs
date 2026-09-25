/**
 * Adapted YuNet postprocessing from OpenCV 4.13.0 FaceDetectorYN:
 * https://github.com/opencv/opencv/blob/4.13.0/modules/objdetect/src/face_detect.cpp
 * OpenCV contributors, Apache License 2.0; see repository LICENSE.
 * This JavaScript translation is for local parity experiments only.
 */
export function decodeYunet(outputs, padW, padH, scoreThreshold = 0.9, nmsThreshold = 0.3) {
  const faces = []
  for (const stride of [8, 16, 32]) {
    const cls = outputs[`cls_${stride}`].data
    const obj = outputs[`obj_${stride}`].data
    const bbox = outputs[`bbox_${stride}`].data
    const kps = outputs[`kps_${stride}`].data
    const cols = padW / stride
    const rows = padH / stride
    if (!Number.isInteger(cols) || !Number.isInteger(rows) || cls.length !== cols * rows || obj.length !== cls.length || bbox.length !== 4 * cls.length || kps.length !== 10 * cls.length) {
      throw new Error(`Unexpected YuNet ${stride} output dimensions`)
    }
    for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
      const index = row * cols + col
      const score = Math.sqrt(Math.max(0, Math.min(1, cls[index])) * Math.min(1, obj[index]))
      if (score < scoreThreshold) continue
      const centerX = (col + bbox[4 * index]) * stride
      const centerY = (row + bbox[4 * index + 1]) * stride
      const width = Math.exp(bbox[4 * index + 2]) * stride
      const height = Math.exp(bbox[4 * index + 3]) * stride
      const face = [centerX - width / 2, centerY - height / 2, width, height]
      for (let point = 0; point < 5; point++) {
        face.push((col + kps[10 * index + 2 * point]) * stride)
        face.push((row + kps[10 * index + 2 * point + 1]) * stride)
      }
      face.push(score)
      faces.push(face)
    }
  }
  // OpenCV performs NMS on integer-truncated XYWH boxes only when >1 candidate.
  if (faces.length < 2) return faces
  const ranked = faces.map((face, index) => ({ face, index }))
    .sort((a, b) => b.face[14] - a.face[14] || a.index - b.index)
  const kept = []
  const overlap = (a, b) => {
    const x0 = Math.max(Math.trunc(a[0]), Math.trunc(b[0]))
    const y0 = Math.max(Math.trunc(a[1]), Math.trunc(b[1]))
    const x1 = Math.min(Math.trunc(a[0]) + Math.trunc(a[2]), Math.trunc(b[0]) + Math.trunc(b[2]))
    const y1 = Math.min(Math.trunc(a[1]) + Math.trunc(a[3]), Math.trunc(b[1]) + Math.trunc(b[3]))
    const area = Math.max(0, x1 - x0) * Math.max(0, y1 - y0)
    const areaA = Math.trunc(a[2]) * Math.trunc(a[3])
    const areaB = Math.trunc(b[2]) * Math.trunc(b[3])
    return area / (areaA + areaB - area)
  }
  for (const candidate of ranked.slice(0, 5000)) {
    if (kept.every(face => overlap(face, candidate.face) <= nmsThreshold)) kept.push(candidate.face)
  }
  return kept
}
