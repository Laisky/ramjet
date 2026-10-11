/** Exercise the checked-in operator form in Chromium with inert HTTP responses. */
import fs from 'node:fs';
import assert from 'node:assert/strict';

const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const template = fs.readFileSync('ramjet/tasks/templates/twitter/index.html', 'utf8');
const body = template.split('{% block body %}')[1]?.split('{% endblock %}')[0];
assert.ok(body, 'actual Twitter form template missing');
const legacy = process.argv.includes('--reproduce-legacy');
const script = legacy ? '' : fs.readFileSync('ramjet/tasks/static/twitter-operator-fetch.js', 'utf8');
const html = legacy ? '<!doctype html><html><body>' + body + '</body></html>' :
    fs.readFileSync(process.env.OPERATOR_HTML_FIXTURE, 'utf8');
const css = legacy ? '' : fs.readFileSync('ramjet/tasks/static/twitter-operator.css', 'utf8');
const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROME_EXECUTABLE || '/usr/bin/google-chrome',
    args: ['--no-sandbox', '--disable-background-networking', '--disable-component-update', '--disable-sync'],
});
let cases = 0;
try {
    const context = await browser.newContext({serviceWorkers: 'block'});
    await context.addCookies([{name: 'session', value: 'inert-existing-cookie', domain: 'app.fixture', path: '/'}]);
    const requests = [];
    let mode = 'ok';
    let releasePending;
    await context.route('**/*', async route => {
        const request = route.request();
        const url = new URL(request.url());
        assert.equal(url.origin, 'https://app.fixture', 'external request attempted');
        if (request.method() === 'POST') {
            requests.push({
                url: request.url(),
                authorization: request.headers()['authorization'],
                body: request.postData(),
                cookies: request.headers()['cookie'],
                referer: request.headers()['referer'],
            });
            if (mode === 'pending') {
                await new Promise(resolve => {releasePending = resolve;});
            }
            if (mode === 'network') {
                await route.abort();
                return;
            }
            const status = legacy ? 401 : mode === 'unconfigured' ? 503 : mode === 'denied' ? 401 : 200;
            await route.fulfill({status, contentType: 'text/plain', body: '<img src=x onerror=alert(1)>'});
        } else if (url.pathname === '/static/twitter-operator.css') {
            await route.fulfill({status: 200, contentType: 'text/css', body: css});
        } else if (url.pathname === '/static/twitter-operator-fetch.js') {
            await route.fulfill({status: 200, contentType: 'text/javascript', body: script});
        } else {
            await route.fulfill({
                status: 200,
                contentType: 'text/html',
                body: html,
            });
        }
    });
    const page = await context.newPage();
    await page.goto('https://app.fixture/prefix/twitter/');
    await page.locator('input[name="tweet_id"]').fill('https://x.com/local/status/12345?fixture=1');
    if (legacy) {
        await Promise.all([
            page.waitForNavigation(),
            page.locator('input[type="submit"]').click(),
        ]);
        assert.equal(requests.length, 1);
        assert.equal(requests[0].authorization, undefined, 'legacy form unexpectedly had bearer');
        assert.ok(requests[0].body.includes('tweet_id'));
        assert.ok(requests[0].body.includes('12345'));
        console.log(JSON.stringify({case: 'legacy-multipart-browser', status: 401, headerPresent: false}));
    } else {
        assert.deepEqual(await page.evaluate(() => [...document.querySelectorAll('script[src],link[href]')]
            .map(element => new URL(element.src || element.href).origin)),
        ['https://app.fixture', 'https://app.fixture']);
        assert.equal(await page.locator('#twitterOperatorToken').evaluate(element =>
            getComputedStyle(element).display), 'block', 'local operator styles missing');
        const token = 'public-local-operator-token';
        const submit = async (nextMode, credential = token) => {
            mode = nextMode;
            await page.locator('#twitterOperatorToken').fill(credential);
            await page.evaluate(() => document.querySelector('#twitterFetch').dispatchEvent(
                new Event('submit', {cancelable: true}),
            ));
        };
        const waitStatus = text => page.waitForFunction(expected =>
            document.querySelector('#twitterFetchStatus').textContent === expected, text);
        await submit('ok');
        await waitStatus('Tweet queued.');
        assert.equal(requests.length, 1);
        assert.equal(requests[0].url, 'https://app.fixture/prefix/twitter/fetch/');
        assert.equal(requests[0].authorization, 'Bearer ' + token);
        assert.ok(requests[0].body.includes('12345'));
        assert.ok(!requests[0].body.includes(token), 'credential entered form payload');
        assert.equal(requests[0].cookies, undefined);
        assert.equal(requests[0].referer, undefined);
        assert.equal(await page.locator('#twitterOperatorToken').inputValue(), '');
        assert.deepEqual(await page.evaluate(() => ({
            local: localStorage.length, session: sessionStorage.length,
            urlContainsCredential: location.href.includes('public-local-operator-token'),
            parsedServerHTML: document.querySelector('#twitterFetchStatus img') !== null,
        })), {local: 0, session: 0, urlContainsCredential: false, parsedServerHTML: false});
        cases += 1;
        await submit('denied');
        await waitStatus('Operator credential rejected.');
        cases += 1;
        await submit('unconfigured');
        await waitStatus('Operator endpoint is not configured.');
        cases += 1;
        await submit('network');
        await waitStatus('Unable to submit tweet. Please try again.');
        cases += 1;
        const before = requests.length;
        await submit('ok', '');
        await waitStatus('Enter an operator credential.');
        assert.equal(requests.length, before);
        cases += 1;
        await page.evaluate(() => {document.querySelector('#twitterFetch').action = 'https://external.fixture/fetch/';});
        await submit('ok');
        await waitStatus('Unable to submit tweet. Please try again.');
        assert.equal(requests.length, before, 'cross-origin action dispatched');
        assert.equal(await page.locator('#twitterOperatorToken').inputValue(), '');
        cases += 1;
        await page.evaluate(() => {document.querySelector('#twitterFetch').action = 'fetch/';});
        await submit('pending');
        await page.waitForFunction(() => document.querySelector('#twitterFetchSubmit').disabled);
        await page.waitForTimeout(50);
        await page.evaluate(() => document.querySelector('#twitterFetch').dispatchEvent(
            new Event('submit', {cancelable: true}),
        ));
        while (!releasePending) {await page.waitForTimeout(10);}
        releasePending();
        await waitStatus('Tweet queued.');
        assert.equal(requests.length, before + 1, 'duplicate pending submission queued');
        assert.equal(await page.locator('#twitterFetchSubmit').isDisabled(), false);
        cases += 1;
        console.log(JSON.stringify({cases, result: 'passed', effects: 'mock HTTP only'}));
    }
    await context.close();
} finally {
    await browser.close();
}
