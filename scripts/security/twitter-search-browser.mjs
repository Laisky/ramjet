/** Run the actual search callback in Chromium using inert local GraphQL data. */
import fs from 'node:fs';
import assert from 'node:assert/strict';

const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const template = fs.readFileSync('ramjet/tasks/templates/twitter/search.html', 'utf8');
const browser = await chromium.launch({
    headless: true, executablePath: process.env.CHROME_EXECUTABLE || '/usr/bin/google-chrome',
    args: ['--no-sandbox'],
});
try {
    const context = await browser.newContext({serviceWorkers: 'block'});
    await context.route('**/*', route => route.abort());
    const page = await context.newPage();
    // Parse trusted checked-in template source without executing it or filtering HTML.
    const callback = await page.evaluate(markup => {
        const document = new DOMParser().parseFromString(markup, 'text/html');
        return [...document.querySelectorAll('script')]
            .find(script => !script.src && script.textContent.includes('twitterSearch'))?.textContent;
    }, template);
    assert.ok(callback, 'search callback missing from the actual template');
    await page.setContent('<form id="twitterSearch"><input value="fixture"></form><article id="tweets"></article>');
    const payload = '<img src="https://invalid.local/fixture" onerror="window.__tweetCanary=(window.__tweetCanary||0)+1">';
    await page.evaluate(({payload}) => {
        window.__tweetCanary = 0;
        window.graphql = {request: async () => ({TwitterStatues: [
            {id: '12/34?x="', text: payload, created_at: payload, user: {name: payload}},
            {id: '456', text: 'normal <literal> & Unicode ☃', created_at: '2026-01-01', user: null},
        ]})};
        // Model the existing jQuery .html() DOM sink without remote CDN loading.
        window.$ = selector => ({html: markup => {document.querySelector(selector).innerHTML = markup;}});
    }, {payload});
    await page.addScriptTag({content: callback});
    await page.evaluate(() => document.querySelector('#twitterSearch').dispatchEvent(new Event('submit', {cancelable: true})));
    await page.waitForTimeout(150);
    const result = await page.evaluate(() => ({
        canary: window.__tweetCanary,
        images: document.querySelectorAll('#tweets img').length,
        rows: document.querySelectorAll('#tweets a').length,
        href: document.querySelector('#tweets a')?.getAttribute('href'),
        text: document.querySelector('#tweets')?.textContent,
    }));
    console.log(JSON.stringify(result));
    assert.equal(result.canary, 0, 'untrusted tweet fields executed an event handler');
    assert.equal(result.images, 0, 'untrusted tweet fields were parsed as HTML');
    assert.equal(result.rows, 2);
    assert.equal(result.href, '/twitter/status/12%2F34%3Fx%3D%22/');
    assert.ok(result.text.includes(payload));
    assert.ok(result.text.includes('normal <literal> & Unicode ☃'));
    assert.ok(result.text.includes('佚名:'));
    await context.close();
} finally {
    await browser.close();
}
