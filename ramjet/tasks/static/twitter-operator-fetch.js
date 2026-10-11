/** Send an operator-authorized tweet request without retaining the credential. */
(() => {
    const form = document.querySelector('#twitterFetch');
    if (!form) {
        return;
    }
    const credential = document.querySelector('#twitterOperatorToken');
    const submit = document.querySelector('#twitterFetchSubmit');
    const status = document.querySelector('#twitterFetchStatus');
    let pending = false;

    // Keep credentials out of form data, URLs, persistent storage, and redirects.
    form.addEventListener('submit', async event => {
        event.preventDefault();
        if (pending) {
            return;
        }
        const token = credential.value;
        credential.value = '';
        if (!token) {
            status.textContent = 'Enter an operator credential.';
            return;
        }
        pending = true;
        submit.disabled = true;
        status.textContent = 'Submitting tweet…';
        try {
            const target = new URL(form.action, window.location.href);
            if (target.origin !== window.location.origin) {
                throw new Error('Operator submission requires the same origin');
            }
            const response = await fetch(target.href, {
                method: 'POST',
                headers: {Authorization: 'Bearer ' + token},
                body: new FormData(form),
                credentials: 'omit',
                redirect: 'error',
                cache: 'no-store',
                referrerPolicy: 'no-referrer',
            });
            status.textContent = response.ok ? 'Tweet queued.' :
                response.status === 401 ? 'Operator credential rejected.' :
                    response.status === 503 ? 'Operator endpoint is not configured.' :
                        'Unable to submit tweet. Please try again.';
        } catch {
            status.textContent = 'Unable to submit tweet. Please try again.';
        } finally {
            pending = false;
            submit.disabled = false;
        }
    });
})();
