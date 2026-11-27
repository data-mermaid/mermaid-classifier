# Training Config: coralnet_all_plus_mermaid_coarse_grouped_minsupport

Generated 2026-11-27 16:31 UTC by
`scripts/generate_training_config.py`. Regenerate with:

```
scripts/generate_training_config.py --output-dir sagemaker/configs/coralnet_all_plus_mermaid_coarse_grouped_minsupport --roll-up-species --roll-up-region-specific --min-support-counts sagemaker/configs/coralnet_all_plus_mermaid_coarse_grouped_minsupport/support_counts.csv --min-points 500 --min-groups 10 --training-config sagemaker/configs/coralnet_all_plus_mermaid_coarse_grouped_minsupport/training_config.yaml
```

`support_counts.csv` (and its provenance sidecar) were generated on 2026-11-26
against the coarse label set, which holds 94 classes with rows, by:

```
scripts/count_class_support.py --config-dir sagemaker/configs/coralnet_all_plus_mermaid_coarse_grouped --out sagemaker/configs/coralnet_all_plus_mermaid_coarse_grouped_minsupport/support_counts.csv
```

## Logic

Replicates the validated EDA notebook
(`data-mermaid/mermaid-segmentation@coralnet_eda/nbs/EDA/coralnet_eda_validated.ipynb`)
in CoralNet -> MERMAID -> top-108 hierarchy-walk space, against the live
MERMAID API. For each CoralNet-mapped benthic attribute:

1. If the BA is in `EXCLUDED_NAMES` (Dead coral, Bleached coral, Other invertebrates), drop entirely.
2. If already in the class set (the top-108 after the roll-up rules, see Coarsened labels below), keep as-is.
3. Otherwise walk parents (nearest first) to find a class-set ancestor.
4. Otherwise drop the label (`LabelFilter` removes it at training time).

## Source-of-truth inputs

| File | Modified | sha1[:10] |
| --- | --- | --- |
| `drive_label_mappings/coralnet_labels_mermaid_mapping_annotations - mapped_to_mermaid_attributes.csv` | 2026-04-28 21:40:41 UTC | `40164622ad` |
| `sagemaker/sources/CoralNetSourcesKept.csv` | 2026-02-17 20:26:28 UTC | `f57b50298e` |
| `sagemaker/configs/coralnet_all_plus_mermaid_coarse_grouped_minsupport/support_counts.csv` | 2026-11-26 23:29:27 UTC | `2b3aece320` |

Live MERMAID API (snapshotted at run time):
- `https://api.datamermaid.org/v1/benthicattributes/?limit=5000` -> sha1 `1ad95d3ef5`
- `https://api.datamermaid.org/v1/classification/labelmappings/?provider=CoralNet` -> sha1 `162f687588`

## Outputs

- `groups.csv` - **327** CoralNet source groups. The generator writes it but it
  is not kept: the training config names no `groups_csv`, so every group trains.
- `included_labels.csv` - **82** `(ba_id, gf_id)` rows
  (Porites contributes 3 rows for the Branching / Massive / other-none buckets;
  every other class-set BA contributes 1 row).
- `rollups.csv` - **260** unique `(from_ba, from_gf)` rows:
    - GF-collapse rows (top-108 BA with non-bucket GF -> same BA, empty GF): 37
    - cross-BA rollups (species or genus -> class-set ancestor): 223

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
- No top-108 ancestor in the parent chain: 103
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

Rules: species, region-specific (exactly one MERMAID region), min support (>= 500 points and >= 10 groups, counts from support_counts.csv, deepest level first, Porites judged with its surviving growth-form buckets).

Classes added as roll-up targets (7): Agariciidae, Astrocoeniidae, Fungiidae, Meandrinidae, Mussidae, Oculinidae, Siderastrea.

Labels whose class changes (53), CoralNet-mapped or top-108:

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
| Caulastraea | Caulastraea | Hard coral |
| Coeloseris | Coeloseris | Agariciidae |
| Colpophyllia natans | Colpophyllia natans | Colpophyllia |
| Ctenactis | Ctenactis | Fungiidae |
| Cycloseris | Cycloseris | Fungiidae |
| Dendrogyra cylindrus | Hard coral | Meandrinidae |
| Dichocoenia stokesii | Dichocoenia stokesii | Meandrinidae |
| Diploria | Diploria | Mussidae |
| Diploria labyrinthiformis | Diploria labyrinthiformis | Mussidae |
| Eusmilia fastigiata | Hard coral | Meandrinidae |
| Favia fragum | Favia fragum | Favia |
| Halomitra | Hard coral | Fungiidae |
| Helioseris cucullata | Hard coral | Agariciidae |
| Isophyllia rigida | Hard coral | Mussidae |
| Isophyllia sinuosa | Hard coral | Mussidae |
| Lithophyllon | Hard coral | Fungiidae |
| Lobactis scutaria | Hard coral | Fungiidae |
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
| Oulastrea | Oulastrea | Hard coral |
| Oxypora | Oxypora | Hard coral |
| Plerogyra | Plerogyra | Hard coral |
| Podabacia | Podabacia | Fungiidae |
| Polyphyllia | Polyphyllia | Fungiidae |
| Porites astreoides | Porites astreoides | Porites |
| Pseudodiploria clivosa | Hard coral | Mussidae |
| Pseudodiploria strigosa | Hard coral | Mussidae |
| Sandalolitha | Sandalolitha | Fungiidae |
| Sea cucumber | Sea cucumber | (dropped) |
| Siderastrea | Hard coral | Siderastrea |
| Siderastrea radians | Hard coral | Siderastrea |
| Siderastrea siderea | Siderastrea siderea | Siderastrea |
| Stephanocoenia intersepta | Stephanocoenia intersepta | Astrocoeniidae |
| Stylocoeniella | Stylocoeniella | Astrocoeniidae |

The support counts' provenance check covers the dataset, group, taxonomy and project-lookup pins and the counts file itself. It does not verify the label-filter, rollup or excluded-images CSVs the counts were made with: regenerate the counts after editing them.

Classes rolled up by min support (16), in roll-up order:

| Class | Points | Groups | Target |
| --- | --- | --- | --- |
| Sandalolitha | 177 | 35 | Fungiidae |
| Ctenactis | 301 | 46 | Fungiidae |
| Podabacia | 68 | 12 | Fungiidae |
| Cycloseris | 42 | 13 | Fungiidae |
| Oxypora | 308 | 35 | Hard coral |
| Polyphyllia | 31 | 4 | Fungiidae |
| Dichocoenia | 299 | 17 | Meandrinidae |
| Stylocoeniella | 118 | 39 | Astrocoeniidae |
| Caulastraea | 206 | 45 | Hard coral |
| Lobophylliidae | 308 | 35 | Hard coral |
| Coeloseris | 15 | 8 | Agariciidae |
| Scleractinia | 308 | 35 | Hard coral |
| Merulinidae | 206 | 45 | Hard coral |
| Sea cucumber | 464 | 63 | (dropped) |
| Plerogyra | 174 | 15 | Hard coral |
| Oulastrea | 22 | 3 | Hard coral |
