# Upload job lifetime regression (#246)

The old handler accepted two unresolved jobs with an existing one-job permit. A bounded two-worker local test also occupied both outer workers and reproduced same-pool embedding starvation. No provider, storage service, or production traffic was used.

The fix holds the existing per-user permit from before multipart parsing until the actual worker Future completes. Success, error, cancellation, and scheduling failures release it once. Background failures are observed. Each worker owns a duplicated uploaded-file descriptor so request cleanup cannot close it early. File orchestration and embedding batches run directly in the occupied worker instead of submitting work to that same executor and waiting.

Provider URLs, caller identity, paid/free entitlements, and per-user capacity retain their existing behavior. Serial embedding batches reduce within-job parallelism while allowing outer jobs to complete.

This is a partial repair: caller-selected identities, the universal concurrent allowance, the free-user semaphore key, globally unbounded admission, and provider task deadlines still require an explicit policy. No global cap or new account policy is introduced by this PR.

Run local regressions with the project's Python environment:
python -m unittest discover -s tests -p test_upload_jobs.py -v
