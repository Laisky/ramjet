# Private chatbot cache password checks

On master 34e2a014a1c61e2e457690c1719de33421b378d6, search/query authenticated the password only when restoring an encrypted chatbot. After a legitimate restore, any nonempty password could retrieve the cached plaintext. A sibling chatbot password also passed against a different active chatbot. Cached selected-dataset metadata had the same gap.

The retained offline regression initially failed five cases against unchanged master: warm wrong-password access, sibling-password access, an unproven plaintext cache entry, cached selected metadata, and the newly built chatbot cache. The other six initial contracts passed. These tests use synthetic encrypted objects, actual AES-EAX authentication and the existing key derivation; storage, vector deserialization and model calls are inert. They do not establish namespace ownership or exercise production.

## Change and compatibility

A process-local AES-EAX marker now binds successful password verification to the exact cached chain. It uses the existing derive_key function (PBKDF2-HMAC-SHA256, 100000 iterations). The cache retains encrypted proof, never the password or derived key. Restored chains and newly built chains install their proof atomically with the existing cache lock.

Search/query and selected private metadata verify that proof before reading a warm chain. Wrong passwords fail authenticated verification as on the cold path. Missing or mismatched proof triggers the existing authenticated cold restore. The returned chain is the authenticated local snapshot, so a concurrent sibling writer cannot substitute another chain after verification. An active sibling chatbot with a different password remains rejected; there is no fallback to a different default chatbot.

Valid warm requests keep the cache and avoid repeat object downloads. UID derivation, storage paths, serialized indexes, current pointers, model behavior and public sharing are unchanged. Empty or invalid passwords leave selected metadata empty, matching cold-path behavior. Dataset-name listing and deletion authorization are outside this compatible patch.

The 13 retained tests cover cold/warm correct and incorrect passwords, sibling passwords, missing headers, unproven/replaced entries, cache-write paths, failed restore preservation, public sharing, serialization and concurrent replacement. Full local suites run offline on Python 3.10 and 3.12. Automatic CI remains the existing formatting and fast-unit gate.

## Remaining ownership boundary

A data password demonstrates an object-specific decrypt capability; it does not establish an exclusive creator or ownership of every object in a UID namespace. A supplied UID, a valid current model API key or a matching truncated gateway ID likewise does not prove namespace-wide ownership. Listing, deletion and mutations require a separate policy decision and verified ownership design. This patch neither migrates namespaces nor suppresses the existing sensitive-hashing finding.
