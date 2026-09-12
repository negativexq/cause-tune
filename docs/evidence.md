# Run evidence and provenance

Every executable training/evaluation run can be initialized as a versioned
evidence bundle:

```text
runs/<run_id>/
├── manifest.json
├── resolved_config.json
├── environment.json
├── data_manifest.json
├── checkpoint_selection.json
├── evaluation_contract.json
└── artifact_hashes.json
```

`run_id` is derived from the experiment contract and content identities of the
train, validation and benchmark data. Absolute paths and output directories
are recorded as metadata but do not change that experiment fingerprint. The
manifest also records the current git commit and dirty state, model revision,
resolved-config SHA-256, data SHA-256 values, configured/actual training steps,
stop reason and validation-only checkpoint selection.

`finalize_evidence()` hashes persisted artifacts by relative name. The manifest
and hash index are excluded from their own artifact set. Finalization is
idempotent when the bundle is unchanged, then seals the bundle: changing an
artifact and attempting to re-finalize fails instead of silently rewriting
provenance. Optional artifacts are simply absent from the index; an adapter,
predictions or evaluation file is hashed when it exists. A completed training
record cannot be finalized without an explicit checkpoint-selection record.
