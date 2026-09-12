# EX-LV — Low-VRAM / Layer Streaming Research Track

EX-LV is not a required CauseTune backend feature. Soup is an external
reference implementation for the first controlled comparison; CauseTune does
not yet ship a streaming engine.

Admission is sequential:

- `GLV0` requires forward-loss, full-gradient, LoRA-gradient and optimizer-
  update parity. Loss parity alone is explicitly insufficient.
- `GLV1` measures the same model/tokenizer/data/sequence length/LoRA/seed/
  optimizer/steps for resident and streamed paths, including VRAM, host RAM,
  disk I/O, wall time, throughput and quality.
- `GLV2` admits a backend as `experimental` only when correctness passes,
  memory reduction is meaningful, quality stays within tolerance and the
  result is repeatable.

Until those gates have real controlled measurements, the status is
`not_admitted` and no streaming implementation is part of the main training
path.
