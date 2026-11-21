# Training Config: coralnet_all_plus_mermaid_coarse

Generated 2026-10-05 18:09 UTC by
`scripts/generate_training_config.py`. Regenerate with:

```
scripts/generate_training_config.py --output-dir sagemaker/configs/coralnet_all_plus_mermaid_coarse --roll-up-species --roll-up-region-specific
```

## Logic

Replicates the validated EDA notebook
(`data-mermaid/mermaid-segmentation@coralnet_eda/nbs/EDA/coralnet_eda_validated.ipynb`)
in CoralNet -> MERMAID -> top-108 hierarchy-walk space, against the live
MERMAID API. For each CoralNet-mapped benthic attribute:

1. If the BA is in `EXCLUDED_NAMES` (Dead coral, Bleached coral, Other invertebrates), drop entirely.
2. If already in the top-108 set, keep as-is.
3. Otherwise walk parents (nearest first) to find a top-108 ancestor.
4. Otherwise drop the label (`LabelFilter` removes it at training time).

## Source-of-truth inputs

| File | Modified | sha1[:10] |
| --- | --- | --- |
| `drive_label_mappings/coralnet_labels_mermaid_mapping_annotations - mapped_to_mermaid_attributes.csv` | 2026-04-28 21:40:41 UTC | `40164622ad` |
| `sagemaker/sources/CoralNetSourcesKept.csv` | 2026-02-17 20:26:28 UTC | `f57b50298e` |

Live MERMAID API (snapshotted at run time):
- `https://api.datamermaid.org/v1/benthicattributes/?limit=5000` -> sha1 `b11f2da182`
- `https://api.datamermaid.org/v1/classification/labelmappings/?provider=CoralNet` -> sha1 `162f687588`

## Outputs

- `sources.csv` - **327** CoralNet source IDs (pass-through; no filtering).
- `included_labels.csv` - **94** `(ba_id, gf_id)` rows
  (Porites contributes 3 rows for the Branching / Massive / other-none buckets;
  every other top-108 BA contributes 1 row).
- `rollups.csv` - **248** unique `(from_ba, from_gf)` rows:
    - GF-collapse rows (top-108 BA with non-bucket GF -> same BA, empty GF): 37
    - cross-BA rollups (species or genus -> top-108 ancestor): 211

### Porites buckets

Per Iain/Emily (2026-05-08), Porites is the only genus that retains
growth-form distinctions. `included_labels.csv` has three Porites rows:
`(Porites, Branching)`, `(Porites, Massive)`, `(Porites, '')`.

Rollups feeding the buckets:
- Porites genus with non-Branching/non-Massive GF -> `(Porites, '')`: 2
- Porites species -> `(Porites, Branching)`: 5
- Porites species -> `(Porites, Massive)`: 7
- Porites species -> `(Porites, '')` (other/no inherent GF): 2

Species' inherent GF comes from the labels CSV's `growth forms` column
(single-value rows only). CN-supplied GF wins over the inherent GF when
both are present.

### Excluded labels

These labels are excluded from training entirely (no rollup, no class):
- `Dead coral`
- `Bleached coral`
- `Other invertebrates`

CoralNet mappings dropped because their BA is in EXCLUDED_NAMES: 67.

### Other dropped CoralNet mappings

- Source BA UUID not in MERMAID API: 0
- No top-108 ancestor in the parent chain: 101
- CoralNet mapping with null benthic_attribute_id: 0

## Deviation from notebook (justified)

The notebook works on benthic-attribute *names* only and ignores growth
forms. The training pipeline's `LabelRollupSpec` does literal
`(from_ba_id, from_gf_id)` tuple lookup, so to keep this config
self-contained (i.e. usable without setting
`DatasetOptions.drop_growthforms=True`) this script enumerates every
`(BA, GF)` pair present in the live CoralNet->MERMAID mapping and emits
one rollup row per pair. Most pairs collapse to `(top-108 BA, '')`;
Porites is the only exception (see above).

## Coarsened labels

Rules: species, region-specific (exactly one MERMAID region).

Classes added as roll-up targets (7): Agariciidae, Astrocoeniidae, Dichocoenia, Meandrinidae, Mussidae, Oculinidae, Siderastrea.

Labels whose class changes (38), CoralNet-mapped or top-108:

| Label | Class without rules | Class with rules |
| --- | --- | --- |
| Acanthaster planci | Acanthaster planci | (dropped) |
| Acropora cervicornis | Acropora cervicornis | Acropora |
| Acropora palmata | Acropora palmata | Acropora |
| Agaricia | Agaricia | Agariciidae |
| Agaricia agaricites | Agaricia agaricites | Agariciidae |
| Agaricia fragilis | Agaricia fragilis | Agariciidae |
| Agaricia grahamae | Agaricia | Agariciidae |
| Agaricia humilis | Agaricia humilis | Agariciidae |
| Agaricia lamarcki | Agaricia | Agariciidae |
| Agaricia tenuifolia | Agaricia tenuifolia | Agariciidae |
| Agariciidae | Hard coral | Agariciidae |
| Colpophyllia natans | Colpophyllia natans | Colpophyllia |
| Dendrogyra cylindrus | Hard coral | Meandrinidae |
| Dichocoenia stokesii | Dichocoenia stokesii | Dichocoenia |
| Diploria | Diploria | Mussidae |
| Diploria labyrinthiformis | Diploria labyrinthiformis | Mussidae |
| Eusmilia fastigiata | Hard coral | Meandrinidae |
| Favia fragum | Favia fragum | Favia |
| Helioseris cucullata | Hard coral | Agariciidae |
| Isophyllia rigida | Hard coral | Mussidae |
| Isophyllia sinuosa | Hard coral | Mussidae |
| Madracis decactis | Madracis decactis | Madracis |
| Meandrina meandrites | Meandrina meandrites | Meandrinidae |
| Montastraea cavernosa | Montastraea cavernosa | Montastraea |
| Mycetophyllia | Mycetophyllia | Mussidae |
| Mycetophyllia ferox | Mycetophyllia | Mussidae |
| Mycetophyllia lamarckiana | Mycetophyllia | Mussidae |
| Orbicella | Orbicella | Oculinidae |
| Orbicella annularis | Orbicella annularis | Oculinidae |
| Orbicella faveolata | Orbicella faveolata | Oculinidae |
| Orbicella franksi | Orbicella franksi | Oculinidae |
| Porites astreoides | Porites astreoides | Porites |
| Pseudodiploria clivosa | Hard coral | Mussidae |
| Pseudodiploria strigosa | Hard coral | Mussidae |
| Siderastrea | Hard coral | Siderastrea |
| Siderastrea radians | Hard coral | Siderastrea |
| Siderastrea siderea | Siderastrea siderea | Siderastrea |
| Stephanocoenia intersepta | Stephanocoenia intersepta | Astrocoeniidae |
