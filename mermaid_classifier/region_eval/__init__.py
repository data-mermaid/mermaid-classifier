"""Region-mismatch measurement: how often a model applies a label from the
wrong region, and how much of that is the model rather than the label data.

`region_rules`, `metrics`, `triage` and `decisions` are the pure computation:
arithmetic and statistics over arrays of points, with no I/O, no network, and
no read of the live benthic-attribute library -- the region map arrives from
the caller as a frozen snapshot, so a curation change upstream cannot move a
model's score for reasons unrelated to the model. `region_rules` holds the
region predicates and the cluster-bootstrap statistics core; `metrics` builds
on them for the out-of-region rates, their denominators, and the per-region,
per-label, per-direction, and confusion tables a report reads; `triage` sorts
each incident into a model error, a suspect region list, or an unrecorded
one; `decisions` prices the mitigations a rate alone cannot choose between --
a region-blind permutation baseline, a masking counterfactual, within-branch
share, and confidence stratification.

`probe_set` builds and freezes the probe itself: the sampled points, the
benthic-attribute region map, display names, taxonomic ancestry, corpus-wide
annotation counts, and the manifest that pins what was frozen. `features`
fetches each probe point's feature vector from S3 into a cache keyed by
`(row, col)` rather than position, and reads that cache back for scoring.

`score` runs a published model artifact over the frozen probe, and `report`
renders what it found into the summary and decisions CSVs and the manifest
JSON that `write_report` writes to a local directory.
"""
