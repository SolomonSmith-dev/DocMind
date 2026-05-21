/**
 * E2E happy path: login → upload a PDF → ask a question → see an answer.
 *
 * Requires:
 *   - DocMind API running at localhost:8000 (DOCMIND_OLLAMA_BASE_URL set)
 *   - `npm run dev` (started automatically by playwright.config.ts)
 */
import { expect, test } from "@playwright/test";
import path from "path";

const FIXTURE_PDF = path.join(__dirname, "fixtures", "sample.pdf");

test.describe("DocMind upload + query happy path", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("login with demo credentials", async ({ page }) => {
    await page.fill('input[autocomplete="username"]', "demo");
    await page.fill('input[autocomplete="current-password"]', "demo1234");
    await page.click('button[type="submit"]');
    await expect(page.locator("text=DocMind")).toBeVisible();
    await expect(page.locator("text=Sign out")).toBeVisible();
  });

  test("upload PDF and ask a question", async ({ page }) => {
    // Login
    await page.fill('input[autocomplete="username"]', "demo");
    await page.fill('input[autocomplete="current-password"]', "demo1234");
    await page.click('button[type="submit"]');
    await expect(page.locator("text=Sign out")).toBeVisible();

    // Upload via file input (bypasses drag-and-drop for reliability)
    const [fileChooser] = await Promise.all([
      page.waitForEvent("filechooser"),
      page.click('[aria-label="Upload area"]'),
    ]);
    await fileChooser.setFiles(FIXTURE_PDF);

    // Wait for success message
    await expect(page.locator("text=Uploaded")).toBeVisible({ timeout: 15_000 });

    // Ask a question
    await page.fill("textarea", "What is this document about?");
    await page.click('button:has-text("Ask")');

    // Wait for answer to appear (spinner disappears, result box appears)
    await expect(page.locator("text=confidence")).toBeVisible({ timeout: 30_000 });
  });
});
