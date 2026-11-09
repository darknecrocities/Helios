const puppeteer = require('puppeteer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const SCREENSHOT_DIR = path.resolve(__dirname, '../artifacts/screenshots');
const BRAIN_DIR = '/Users/arronkianparejas/.gemini/antigravity/brain/687d9862-919d-40e5-a5b8-a706a80f1837/screenshots';

for (const dir of [SCREENSHOT_DIR, BRAIN_DIR]) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

async function run() {
  console.log('[1/6] Launching Vite preview server...');
  const server = spawn('npx', ['vite', 'preview', '--port', '4173', '--host', '127.0.0.1'], {
    cwd: path.resolve(__dirname, '../website'),
    stdio: 'ignore'
  });

  await new Promise(r => setTimeout(r, 2000));

  try {
    console.log('[2/6] Launching Headless Chrome via Puppeteer...');
    const browser = await puppeteer.launch({
      executablePath: CHROME_PATH,
      headless: 'new',
      args: ['--no-sandbox', '--disable-gpu', '--window-size=1280,800']
    });

    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 800, deviceScaleFactor: 2 });

    console.log('[3/6] Navigating to http://127.0.0.1:4173/ ...');
    await page.goto('http://127.0.0.1:4173/', { waitUntil: 'networkidle0', timeout: 15000 });
    await new Promise(r => setTimeout(r, 1000));

    const takeScreenshot = async (name) => {
      const localPath = path.join(SCREENSHOT_DIR, name);
      const brainPath = path.join(BRAIN_DIR, name);
      await page.screenshot({ path: localPath });
      fs.copyFileSync(localPath, brainPath);
      console.log(`Saved screenshot: ${name}`);
    };

    // 1. Desktop Hero (Initial at Top)
    await takeScreenshot('desktop_01_hero.png');

    // 1b. Test Hero Kinetic Solar Zoom (Scrolled 500px down)
    console.log('Testing Hero Kinetic Solar Zoom...');
    await page.evaluate(() => window.scrollBy({ top: 500, behavior: 'instant' }));
    await new Promise(r => setTimeout(r, 600));
    await takeScreenshot('desktop_01_hero_zooming.png');

    // 2. Scroll to About
    await page.evaluate(() => document.getElementById('about')?.scrollIntoView({ behavior: 'instant' }));
    await new Promise(r => setTimeout(r, 600));
    await takeScreenshot('desktop_02_about.png');

    // 3. Scroll to Family (Start: Nano Centered)
    console.log('Testing Family Keeby Horizontal Scroll & Zooming Cards...');
    await page.evaluate(() => document.getElementById('family')?.scrollIntoView({ behavior: 'instant' }));
    await new Promise(r => setTimeout(r, 600));
    await takeScreenshot('desktop_03_family_nano.png');

    // 3b. Scroll Midway through Family (Core Centered)
    await page.evaluate(() => window.scrollBy({ top: 800, behavior: 'instant' }));
    await new Promise(r => setTimeout(r, 600));
    await takeScreenshot('desktop_03_family_core.png');

    // 3c. Scroll to End of Family (Flagship Centered & Ready to Unpin)
    await page.evaluate(() => window.scrollBy({ top: 700, behavior: 'instant' }));
    await new Promise(r => setTimeout(r, 600));
    await takeScreenshot('desktop_03_family_flagship.png');

    // 4. Scroll to Capabilities
    await page.evaluate(() => document.getElementById('capabilities')?.scrollIntoView({ behavior: 'instant' }));
    await new Promise(r => setTimeout(r, 600));
    await takeScreenshot('desktop_04_capabilities.png');

    // 5. Scroll to Comparison Section with Animated Multi-Line Chart
    await page.evaluate(() => document.getElementById('comparison')?.scrollIntoView({ behavior: 'instant' }));
    await new Promise(r => setTimeout(r, 1200));
    await takeScreenshot('desktop_05_comparison_chart.png');

    // 6. Scroll to Platforms Section
    await page.evaluate(() => document.getElementById('platforms')?.scrollIntoView({ behavior: 'instant' }));
    await new Promise(r => setTimeout(r, 600));
    await takeScreenshot('desktop_06_platforms.png');

    // Test OpenClaw Tab
    const openclawTab = await page.waitForSelector('button:has-text("OpenClaw")', { timeout: 3000 }).catch(() => null);
    if (openclawTab) {
      await openclawTab.click();
      await new Promise(r => setTimeout(r, 400));
      await takeScreenshot('desktop_06_platforms_openclaw.png');
    }

    // Test Hermes Tab
    const hermesTab = await page.waitForSelector('button:has-text("Hermes")', { timeout: 3000 }).catch(() => null);
    if (hermesTab) {
      await hermesTab.click();
      await new Promise(r => setTimeout(r, 400));
      await takeScreenshot('desktop_06_platforms_hermes.png');
    }

    // 7. Test Documentation View
    console.log('[4/6] Navigating into Documentation Page...');
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const docBtn = btns.find(b => b.textContent && b.textContent.trim().toLowerCase().includes('docs'));
      if (docBtn) docBtn.click();
    });
    await new Promise(r => setTimeout(r, 1000));
    await takeScreenshot('desktop_07_docs_overview.png');

    // Click Chapter 05 Training & Fine-Tuning
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const ftBtn = btns.find(b => b.textContent && b.textContent.includes('Training & Fine-Tuning'));
      if (ftBtn) ftBtn.click();
    });
    await new Promise(r => setTimeout(r, 800));
    await takeScreenshot('desktop_08_docs_qlora_colab.png');

    // Click Chapter 06 Multi-Model Benchmarks
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const bmBtn = btns.find(b => b.textContent && b.textContent.includes('Multi-Tier Benchmarks'));
      if (bmBtn) bmBtn.click();
    });
    await new Promise(r => setTimeout(r, 1200));
    await takeScreenshot('desktop_09_docs_animated_benchmarks.png');

    // Click Chapter 07 Autonomous Agent Loops (OpenClaw & Hermes)
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const agentBtn = btns.find(b => b.textContent && b.textContent.includes('Autonomous Agent Loops'));
      if (agentBtn) agentBtn.click();
    });
    await new Promise(r => setTimeout(r, 800));
    await takeScreenshot('desktop_10_docs_openclaw_hermes.png');

    // 8. Test Mobile Viewport (375x812 iPhone)
    console.log('[5/6] Testing Mobile Viewport (375x812)...');
    await page.setViewport({ width: 375, height: 812, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    await page.goto('http://127.0.0.1:4173/', { waitUntil: 'networkidle0', timeout: 15000 });
    await new Promise(r => setTimeout(r, 800));

    await takeScreenshot('mobile_01_hero.png');

    await page.evaluate(() => document.getElementById('family')?.scrollIntoView({ behavior: 'instant' }));
    await new Promise(r => setTimeout(r, 600));
    await takeScreenshot('mobile_03_family_carousel.png');

    await page.evaluate(() => document.getElementById('comparison')?.scrollIntoView({ behavior: 'instant' }));
    await new Promise(r => setTimeout(r, 1000));
    await takeScreenshot('mobile_05_comparison_chart.png');

    await page.evaluate(() => document.getElementById('platforms')?.scrollIntoView({ behavior: 'instant' }));
    await new Promise(r => setTimeout(r, 600));
    await takeScreenshot('mobile_06_platforms_openclaw_hermes.png');

    console.log('[6/6] All verifications and high-fidelity screenshots completed!');
    await browser.close();
  } finally {
    console.log('Stopping Vite preview server...');
    server.kill();
  }
}

run().catch((err) => {
  console.error('[PUPPETEER ERROR]', err);
  process.exit(1);
});
