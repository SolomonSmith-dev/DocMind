import { expect, test } from '@playwright/test'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const samplePdf = path.resolve(__dirname, 'fixtures/sample.pdf')

test('upload and ask flow', async ({ page }) => {
  await page.goto('/')

  await page.setInputFiles('input[type="file"]', samplePdf)
  await expect(page.getByText('sample.pdf')).toBeVisible({ timeout: 20000 })

  await page.getByLabel('Question').fill('What is this document about?')
  await page.getByRole('button', { name: 'Ask DocMind' }).click()

  const answer = page.locator('.answer')
  await expect(answer).not.toContainText(
    'Ask a question to see a cited response.',
    { timeout: 30000 },
  )
  await expect(page.getByRole('heading', { name: 'Citations' })).toBeVisible()
})
