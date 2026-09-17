"""Region-mismatch measurement: how often a model applies a label from the
wrong region, and how much of that is the model rather than the label data.

`metrics` computes the out-of-region rates, their denominators, and the
per-region, per-label, and per-direction tables that explain them. `triage`
sorts each incident into a model error, a suspect region list, or an
unrecorded one. `decisions` holds the statistics that price the mitigations a
rate alone cannot choose between: a region-blind permutation baseline, a
masking counterfactual, within-branch share, and confidence stratification.
These three are pure formulas over arrays — nothing in them reaches S3,
MLflow, a model file, or a plotting backend, and nothing reads the live
benthic-attribute library: the region map arrives from the caller as a frozen
snapshot, so a curation change upstream cannot move a model's score for
reasons unrelated to the model.

`probe_set` builds and freezes the probe itself: the sampled points, the
benthic-attribute region map, display names, taxonomic ancestry, corpus-wide
annotation counts, and the manifest that pins what was frozen. `features`
fetches and caches each probe point's feature vector from S3, matched by
`(row, col)` rather than position. `score` runs a published model artifact
over the frozen probe and writes the report.
"""
