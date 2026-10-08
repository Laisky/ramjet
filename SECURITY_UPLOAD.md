Prototype upload bounds
=======================

The application request-body limit is 100 MiB, matching the previous comment's intended bound. Prototype ZIPs are additionally limited to 100 MiB compressed, 500 MiB expanded, 10,000 entries and a 1,000:1 per-entry compression ratio. Extraction checks a cooperative 30-second deadline between bounded reads. These bounds are local constants in the upload handler; oversized prototypes must be split or the limits reviewed deliberately.

Only one prototype upload per application process is admitted at a time. Concurrent requests receive 429. The slot stays occupied until extraction finishes even if the HTTP waiter disconnects. This bounds this upload path, not unrelated executor users or uploads across multiple worker processes.

ZIP metadata and decoded filenames are checked before extraction. Absolute, traversing, Windows-drive, backslash, symlink, encrypted, duplicate and malformed entries are rejected. Standard UTF-8 names and existing legacy GBK names remain supported. Rejected input leaves the existing prototype untouched. Publication retains the existing directory replacement behavior after successful extraction.

Run retained local tests with `python -m unittest tests.test_upload_bounds -v`.

The tests use tiny synthetic archives, local temporary destinations and one bounded worker. They cover extraction/body limits, unsafe metadata, encoding compatibility, rejected-input preservation and cancellation/admission lifetime. They make no production upload or external network request.
