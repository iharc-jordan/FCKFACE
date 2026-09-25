// Android Chrome workflow check. Run only against a task-owned emulator instance.
const { _android } = require('playwright-core');
const { mkdir, writeFile } = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');

async function fixture(page) {
  const base64 = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 640; canvas.height = 480;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#718391'; ctx.fillRect(0, 0, 640, 480);
    ctx.fillStyle = '#b3c678'; ctx.fillRect(0, 0, 320, 240);
    return canvas.toDataURL('image/jpeg', .95).split(',')[1];
  });
  const jpeg = Buffer.from(base64, 'base64');
  const segment = (marker, content) => {
    const header = Buffer.from([255, marker, 0, 0]);
    header.writeUInt16BE(content.length + 2, 2);
    return Buffer.concat([header, content]);
  };
  const exif = Buffer.from('4578696600004d4d002a00000008000101120003000000010006000000000000', 'hex');
  return Buffer.concat([jpeg.subarray(0, 2), segment(225, exif),
    segment(254, Buffer.from('PRIVATE_SOURCE_METADATA')), jpeg.subarray(2)]);
}

async function main() {
  const [serial, output] = process.argv.slice(2);
  assert(serial && output, 'Usage: node research/browser/android_workflow.cjs SERIAL OUTPUT_DIRECTORY');
  await mkdir(output, { recursive: true });
  const report = { started: new Date().toISOString(), serial, scope: 'Unvalidated dot-preview workflow with a synthetic fixture; programmatic file selection, not OS picker or protection validation.' };
  let device, context, page;
  try {
    const devices = await _android.devices({ omitDriverInstall: true });
    device = devices.find(item => item.serial() === serial);
    assert(device, `The requested emulator ${serial} is not connected`);
    device.setDefaultTimeout(30000);
    report.model = device.model();
    assert.equal((await device.shell('test ! -e /sdcard/Download/fckface-research.jpg && echo ABSENT')).toString().trim(), 'ABSENT', 'Use a clean task-owned download location; an existing file could produce a false pass');
    let launchTimer;
    try {
      context = await Promise.race([
        device.launchBrowser({ acceptDownloads: true }),
        new Promise((_, reject) => { launchTimer = setTimeout(() => reject(new Error('Chrome launch exceeded 30 seconds; check Android command-line flag setup')), 30000); }),
      ]);
    } finally { clearTimeout(launchTimer); }
    context.setDefaultTimeout(30000);
    page = await context.newPage();
    const external = [], sent = [], errors = [];
    page.on('request', request => {
      if (/^https?:/.test(request.url()) && !request.url().startsWith('http://127.0.0.1:5173/')) external.push(request.url());
      if (request.postDataBuffer()?.length) sent.push(request.url());
    });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:5173/');
    await page.getByRole('heading', { name: 'Image effects under examination.' }).waitFor();
    await device.shell('uiautomator dump /sdcard/fckface-workflow-ui.xml');
    report.environment = await page.evaluate(() => ({ userAgent: navigator.userAgent, width: innerWidth, height: innerHeight, dpr: devicePixelRatio, secure: isSecureContext }));
    await writeFile(path.join(output, 'entry-device.png'), await device.screenshot());
    await page.locator('input[type=file]').setInputFiles({ name: 'synthetic-private-source.jpg', mimeType: 'image/jpeg', buffer: await fixture(page) });
    await page.getByRole('button', { name: 'Start with centered region' }).click();
    await page.getByText('480 × 640 px', { exact: false }).first().waitFor();
    await device.shell('uiautomator dump /sdcard/fckface-workflow-ui.xml');
    await writeFile(path.join(output, 'selected-device.png'), await device.screenshot());
    const started = Date.now();
    await page.getByRole('button', { name: 'Apply dot preview' }).click();
    const downloadLink = page.getByRole('link', { name: 'Download JPEG' });
    await downloadLink.waitFor();
    report.previewMilliseconds = Date.now() - started;
    report.dimensions = await page.getByAltText('Full photo with artificial dots over the selected face').evaluate(image => [image.naturalWidth, image.naturalHeight]);
    assert.deepEqual(report.dimensions, [480, 640]);
    report.layoutFits = await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth);
    assert(report.layoutFits, 'Layout exceeds viewport');
    await page.getByAltText('Full photo with artificial dots over the selected face').scrollIntoViewIfNeeded();
    await device.shell('uiautomator dump /sdcard/fckface-workflow-ui.xml');
    await writeFile(path.join(output, 'result-device.png'), await device.screenshot());
    Object.assign(report, { externalRequests: external, requestsWithBody: sent, pageErrors: errors });
    assert.equal(await downloadLink.getAttribute('download'), 'fckface-research.jpg');
    // Android Chrome saves through Android's download manager, which does not
    // reliably emit Playwright's desktop download event. Inspect the saved file.
    await downloadLink.click();
    let bytes;
    const deadline = Date.now() + 15000;
    while (Date.now() < deadline) {
      bytes = await device.shell('cat /sdcard/Download/fckface-research.jpg');
      if (bytes.subarray(0, 2).toString('hex') === 'ffd8') break;
      await new Promise(resolve => setTimeout(resolve, 250));
    }
    await writeFile(path.join(output, 'result.jpg'), bytes);
    assert.equal(bytes.subarray(0, 2).toString('hex'), 'ffd8');
    assert(!bytes.includes(Buffer.from('PRIVATE_SOURCE_METADATA')));
    assert(!bytes.includes(Buffer.from('Exif\0\0')));
    assert.deepEqual(external, []); assert.deepEqual(sent, []); assert.deepEqual(errors, []);
    Object.assign(report, { status: 'passed', downloadedBytes: bytes.length, sourceMetadataRemoved: true, externalRequests: external, requestsWithBody: sent, pageErrors: errors });
  } catch (error) {
    Object.assign(report, { status: 'failed', error: String(error.stack || error) });
    if (page) await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true }).catch(() => {});
    if (device) await writeFile(path.join(output, 'device-failure.png'), await device.screenshot()).catch(() => {});
    process.exitCode = 1;
  } finally {
    if (context) await context.close().catch(() => {});
    if (device) await device.close().catch(() => {});
    report.finished = new Date().toISOString();
    await writeFile(path.join(output, 'report.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report, null, 2));
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
