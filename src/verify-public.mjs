import { readdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'

const files = await readdir('dist/assets')
const scripts = files.filter(file => file.endsWith('.js'))
const content = (await Promise.all(scripts.map(file => readFile(join('dist/assets', file), 'utf8')))).join('\n')

if (files.some(file => file.includes('worker')) ||
    content.includes('graphic-dots-preview') ||
    content.includes('Choose a photo') ||
    !content.includes('No photo processor is available here yet')) {
  throw new Error('Public build contains research processing or is missing its status page.')
}
