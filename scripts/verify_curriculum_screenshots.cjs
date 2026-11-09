const path = require('path');
const puppeteer = require(path.resolve(__dirname, '../website/node_modules/puppeteer'));
const { spawn } = require('child_process');
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
  console.log('[1/4] Launching Vite preview server...');
  const server = spawn('npx', ['vite', 'preview', '--port', '4173', '--host', '127.0.0.1'], {
    cwd: path.resolve(__dirname, '../website'),
    stdio: 'ignore'
  });

  await new Promise(r => setTimeout(r, 2000));

  try {
    console.log('[2/4] Launching Headless Chrome via Puppeteer...');
    const browser = await puppeteer.launch({
      executablePath: CHROME_PATH,
      headless: 'new',
      args: ['--no-sandbox', '--disable-gpu', '--window-size=1280,1000']
    });

    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 1000, deviceScaleFactor: 2 });

    console.log('[3/4] Navigating to http://127.0.0.1:4173/ ...');
    await page.goto('http://127.0.0.1:4173/', { waitUntil: 'networkidle0', timeout: 15000 });
    await new Promise(r => setTimeout(r, 1000));

    const takeScreenshot = async (name) => {
      const localPath = path.join(SCREENSHOT_DIR, name);
      const brainPath = path.join(BRAIN_DIR, name);
      await page.screenshot({ path: localPath });
      fs.copyFileSync(localPath, brainPath);
      console.log(`Saved screenshot: ${name}`);
    };

    // Scroll to #training
    console.log('Scrolling to #training section...');
    await page.evaluate(() => {
      const el = document.getElementById('training');
      if (el) el.scrollIntoView({ behavior: 'instant', block: 'start' });
    });
    // Wait for count-up animations to settle
    await new Promise(r => setTimeout(r, 2200));
    await takeScreenshot('desktop_curriculum_datasets.png');

    // Scroll down inside #training to view platform stage in full
    await page.evaluate(() => window.scrollBy({ top: 1400, behavior: 'instant' }));
    await new Promise(r => setTimeout(r, 1200));
    await takeScreenshot('desktop_curriculum_platforms.png');

    // Switch to Mobile Viewport (375x812)
    console.log('Testing Mobile Viewport (375x812)...');
    await page.setViewport({ width: 375, height: 812, deviceScaleFactor: 2 });
    await page.evaluate(() => {
      const el = document.getElementById('training');
      if (el) el.scrollIntoView({ behavior: 'instant', block: 'start' });
    });
    await new Promise(r => setTimeout(r, 1500));
    await takeScreenshot('mobile_curriculum_datasets.png');

    await page.evaluate(() => window.scrollBy({ top: 500, behavior: 'instant' }));
    await new Promise(r => setTimeout(r, 1000));
    await takeScreenshot('mobile_curriculum_numbers.png');

    await page.evaluate(() => window.scrollBy({ top: 700, behavior: 'instant' }));
    await new Promise(r => setTimeout(r, 1000));
    await takeScreenshot('mobile_curriculum_platforms.png');

    await browser.close();
    console.log('[4/4] Screenshots captured successfully.');
  } finally {
    server.kill();
  }
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});
