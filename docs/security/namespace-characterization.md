Historical selector observations are superseded by [the private capability repair](private-credential-capabilities.md). Current opt-in controls assert complete-key selection and conservative admission; original receipts retain their source revisions.

# Namespace scope characterization (task20261008-26)

This opt-in evidence records selectors and false-positive controls, not a repair,
security acceptance contract, deployed bypass, or a new ownership policy.

Run locally with RAMJET_NAMESPACE_CHARACTERIZATION=1 and
python -m unittest discover -s tests/manual -p test_namespace_scope_characterization.py -v.

The public request selector receives synthetic keys and a mocked permission
constructor. Assertions cover the constructor arguments selected from UID and
isFree headers; they do not establish the real deployed constructor's validation.
Existing retained investigation77d1d8e separately tested a reviewed class-only
fixture and actual list/delete decorators. No private fixture is copied here.

Semaphore controls distinguish absent versus preseeded public buckets, same
versus distinct UID, and paid versus free selection. Missing keys are rejected.
No provider, object storage, database, configuration module or account is used.

Preserve legacy names. Authoritative principal-to-object/operation enrollment
remains undefined. These tests deliberately do not supply that missing mapping.
The historical warm-cache password finding was independently repaired by PR288;
nothing here claims that old result persists on current merged master.
