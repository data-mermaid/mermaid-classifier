"""Region-mismatch measurement: how often a model applies a label from the
wrong region, and how much of that is the model rather than the label data.

Pure formulas over arrays. Nothing here reaches S3, MLflow, a model file or a
plotting backend, and nothing here reads the live benthic-attribute library:
the region map arrives from the caller as a frozen snapshot, so a curation
change upstream cannot move a model's score for reasons unrelated to the
model.

`metrics` computes the rates and tables; `triage` sorts the individual events
into a suspect region list, an unrecorded one, or a model error.
"""
