#!/usr/bin/env node
/**
 * Production smoke test for the hosted Grafial Composer.
 * Loads COMPOSER_URL (default https://grafial.iridae.com/), waits for WASM
 * boot, loads the minimal example, runs MinimalFlow, and asserts a metric.
 */

import { chromium } from "playwright";

const url = process.env.COMPOSER_URL || "https://grafial.iridae.com/";

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (err) => errors.push(String(err)));
page.on("console", (msg) => {
  if (msg.type() === "error") errors.push(msg.text());
});

try {
  await page.goto(url, { waitUntil: "networkidle", timeout: 120_000 });

  const bootError = page.locator("#boot-error");
  if (await bootError.isVisible().catch(() => false)) {
    throw new Error("Composer boot error is visible — WASM failed to initialize");
  }

  // Wait for WASM init (About / status build identity appears once ready).
  await page.waitForFunction(
    () => {
      const el = document.getElementById("status-build");
      return el && el.textContent && el.textContent.trim().length > 0;
    },
    { timeout: 120_000 },
  );

  // Load minimal example if the dropdown exists.
  const exampleSelect = page.locator("#sel-example");
  if (await exampleSelect.count()) {
    await exampleSelect.selectOption("minimal").catch(async () => {
      // Some deployments may use basename without extension already selected via options.
      const options = await exampleSelect.locator("option").allTextContents();
      const match = options.find((o) => o.includes("minimal"));
      if (match) await exampleSelect.selectOption({ label: match });
    });
    await page.waitForTimeout(500);
  }

  // Prefer MinimalFlow when present.
  const flowSelect = page.locator("#sel-flow");
  await page.waitForFunction(
    () => {
      const sel = document.getElementById("sel-flow");
      return sel && sel.options && sel.options.length > 0;
    },
    { timeout: 60_000 },
  );
  const flowOptions = await flowSelect.locator("option").allTextContents();
  const preferred =
    flowOptions.find((o) => o.includes("MinimalFlow")) || flowOptions.find((o) => o.trim());
  if (!preferred) throw new Error("No flows available after load");
  await flowSelect.selectOption({ label: preferred.trim() });

  await page.locator("#btn-run").click();
  await page.waitForSelector("#pane-results .results-block, #pane-results table, #pane-results .metric", {
    timeout: 60_000,
  });

  const resultsText = await page.locator("#pane-results").innerText();
  if (!/\d/.test(resultsText)) {
    throw new Error(`Expected numeric metric/result content, got:\n${resultsText.slice(0, 500)}`);
  }

  if (errors.length) {
    console.warn("Console/page errors (non-fatal if results present):", errors.slice(0, 5));
  }

  console.log("composer smoke ok:", url, "flow=", preferred.trim());
} finally {
  await browser.close();
}
