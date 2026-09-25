import { expect, test, type Page } from '@playwright/test'
import { readFile } from 'node:fs/promises'

async function fixture(page: Page) {
  const base64 = await page.evaluate(() => {
    const canvas = document.createElement('canvas')
    canvas.width = 640
    canvas.height = 480
    const ctx = canvas.getContext('2d')!
    ctx.fillStyle = '#718391'; ctx.fillRect(0, 0, 640, 480)
    ctx.fillStyle = '#b3c678'; ctx.fillRect(0, 0, 320, 240)
    return canvas.toDataURL('image/jpeg', .95).split(',')[1]
  })
  const jpeg = Buffer.from(base64, 'base64')
  // Big-endian TIFF: a single Orientation=6 entry, plus an identifying COM.
  const exif = Buffer.from('4578696600004d4d002a00000008000101120003000000010006000000000000', 'hex')
  const comment = Buffer.from('PRIVATE_SOURCE_METADATA')
  const segment = (marker: number, content: Buffer) => {
    const header = Buffer.from([255, marker, 0, 0])
    header.writeUInt16BE(content.length + 2, 2)
    return Buffer.concat([header, content])
  }
  return Buffer.concat([jpeg.subarray(0, 2), segment(225, exif), segment(254, comment), jpeg.subarray(2)])
}

async function choose(page: Page) {
  await page.locator('input[type=file]').setInputFiles({
    name: 'private-source.jpg', mimeType: 'image/jpeg', buffer: await fixture(page),
  })
  await page.getByRole('button', { name: 'Start with centered region' }).click()
}

test('oriented JPEG download strips source metadata and stays on-device', async ({ page }, testInfo) => {
  const external: string[] = []
  const sent: string[] = []
  page.on('request', request => {
    if (/^https?:/.test(request.url()) && !request.url().startsWith('http://127.0.0.1:5173/')) external.push(request.url())
    if (request.postDataBuffer()?.length) sent.push(request.url())
  })
  await page.goto('/')
  await choose(page)
  await expect(page.getByText('480 × 640 px', { exact: false }).first()).toBeVisible()
  const started = Date.now()
  await page.getByRole('button', { name: 'Apply dot preview' }).click()
  await expect(page.getByRole('link', { name: 'Download JPEG' })).toBeVisible()
  expect(Date.now() - started).toBeLessThan(60_000)
  const downloaded = page.waitForEvent('download')
  await page.getByRole('link', { name: 'Download JPEG' }).click()
  const download = await downloaded
  expect(await download.failure()).toBeNull()
  const output = testInfo.outputPath('result.jpg')
  await download.saveAs(output)
  const bytes = await readFile(output)
  expect(bytes.subarray(0, 2).toString('hex')).toBe('ffd8')
  expect(bytes.includes(Buffer.from('PRIVATE_SOURCE_METADATA'))).toBe(false)
  expect(bytes.includes(Buffer.from('Exif\0\0'))).toBe(false)
  const dimensions = await page.getByAltText('Full photo with artificial dots over the selected face').evaluate((image: HTMLImageElement) => [image.naturalWidth, image.naturalHeight])
  expect(dimensions).toEqual([480, 640])
  expect(external).toEqual([])
  expect(sent).toEqual([])
})

test('cancel, repeat processing, and replacement release workers and object URLs', async ({ page }) => {
  await page.addInitScript(() => {
    const state = { urls: new Set<string>(), workers: 0 }
    Object.assign(window, { lifecycle: state })
    const create = URL.createObjectURL.bind(URL)
    const revoke = URL.revokeObjectURL.bind(URL)
    URL.createObjectURL = blob => { const url = create(blob); state.urls.add(url); return url }
    URL.revokeObjectURL = url => { state.urls.delete(url); revoke(url) }
    const NativeWorker = Worker
    window.Worker = class extends NativeWorker {
      ended = false
      constructor(url: string | URL, options?: WorkerOptions) { super(url, options); state.workers++ }
      terminate() { if (!this.ended) { state.workers--; this.ended = true }; super.terminate() }
    }
  })
  await page.goto('/')
  await choose(page)
  let unblock!: () => void
  const blocked = new Promise<void>(resolve => { unblock = resolve })
  await page.route('**/process.worker*', async route => { await blocked; await route.continue().catch(() => {}) }, { times: 1 })
  await page.getByRole('button', { name: 'Apply dot preview' }).click()
  await page.getByRole('button', { name: 'Cancel processing' }).click()
  unblock()
  await expect(page.getByRole('link', { name: 'Download JPEG' })).toHaveCount(0)
  for (let i = 0; i < 3; i++) {
    await page.getByRole('button', { name: 'Apply dot preview' }).click()
    await expect(page.getByRole('link', { name: 'Download JPEG' })).toBeVisible()
  }
  expect(await page.evaluate(() => {
    const state = (window as unknown as { lifecycle: { urls: Set<string>; workers: number } }).lifecycle
    return { urls: state.urls.size, workers: state.workers }
  })).toEqual({ urls: 2, workers: 0 })
  await page.locator('input[type=file]').setInputFiles({ name: 'bad.txt', mimeType: 'text/plain', buffer: Buffer.from('not an image') })
  await expect(page.getByRole('alert')).toHaveText('Choose a JPEG, PNG or WebP image.')
  expect(await page.evaluate(() => (window as unknown as { lifecycle: { urls: Set<string> } }).lifecycle.urls.size)).toBe(0)
})

test('mobile layout stays within the viewport and exposes keyboard selection', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await choose(page)
  await page.getByRole('spinbutton', { name: 'Left %', exact: true }).fill('25')
  await page.getByRole('button', { name: 'Apply dot preview' }).click()
  await expect(page.getByRole('link', { name: 'Download JPEG' })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
})
