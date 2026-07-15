# Model-review annotation app

A tool for evaluating a MERMAID classifier against expert annotations. It samples
held-out CoralNet images, lays down each image's ground-truth points, and hosts them in
[**Label Studio**](https://model-review.datamermaid.org/) so reviewers can label each point
by hand. Their labels are then compared against both the ground truth and the model's
predictions.

**Live app: https://model-review.datamermaid.org/**

---

# Change the image set (the common task)

> This is what you do to put a **new set of images** in front of reviewers. It creates a
> **brand-new** Label Studio project and never touches existing ones. The app is already
> deployed and running — you do **not** need any of the infrastructure steps for this.

### What you need first

1. **AWS access.** Log in with your own profile so the tool can read images/manifest/model
   from S3:
   ```bash
   aws sso login --profile wcs-admin
   export AWS_PROFILE=wcs-admin
   ```
2. **A Label Studio API token.** Open https://model-review.datamermaid.org/, sign in, and
   copy your token from **Account & Settings → Access Token** (or
   `https://model-review.datamermaid.org/api/current-user/token`).
   ```bash
   export LABEL_STUDIO_TOKEN=<your-token>
   ```

### Step 1 — edit one file: `image_set.py`

[`mermaid_classifier/model_review/image_set.py`](image_set.py) is the **single place** that
defines which images (and which model) a review uses. Open it and edit the `IMAGE_SET`
values. **Always change `name`** — each distinct name becomes its own new project.

| Field | What it controls |
| --- | --- |
| `name` | The Label Studio project title (and it's recorded with every annotation). **Change this for each new set.** |
| `classifier` | The model being reviewed. Defaults to `s3://mermaid-config/classifier/v2/` and is **downloaded automatically**. |
| `n_images` | How many images to sample. |
| `min_points` | Only sample images with at least this many ground-truth points. |
| `seed` | Random seed — change it to draw a different random sample of images. |
| `heldout_csv` | The held-out validation split that defines which images are eligible. |
| `manifest_uri` | CoralNet manifest parquet holding the full ground-truth grid per image. |
| `feature_bucket` | Bucket with the per-image feature vectors (used for model inference). |
| `image_bucket` / `image_prefix` / `image_key_template` | Where the display images live in S3. |
| `v1_rollup_csv` | Rollup that maps ground truth into the model's label set. |

For the common case ("same as before, but a different batch of images") you only touch
`name`, `n_images`, and `seed`.

### Step 2 — run one command

```bash
uv run --extra training --with pillow \
  python -m mermaid_classifier.model_review.cli create-project \
  --ls-url https://model-review.datamermaid.org
```

This builds the tasks from `image_set.py` and creates the new project: it attaches the S3
image storage, imports the tasks, and sets reviewers to start **blind** (unlabelled points,
so they aren't biased by the model). It prints the new project URL. It takes ~1 minute
(it downloads the model and reads image sizes from S3).

- The token is read from `LABEL_STUDIO_TOKEN` (or pass `--token <token>`).
- If a project with the same `name` already exists, the command **stops and refuses** —
  change `name` in `image_set.py` and rerun. Existing projects are never modified.

### Step 3 — reviewers annotate

Each reviewer signs in with **their own account** (annotations are attributed to whoever
labelled them). They open each image to grey, unlabelled fixed points and assign the fine
benthic-attribute label to each via the taxonomy tree. They can toggle read-only
`ground-truth` and `v1` reference layers *after* their pass, and use the per-image notes box.

> Images are served straight from S3 to the browser. This relies on a one-time CORS rule on
> the image bucket for the app's origin (normally already configured for the hosted app);
> see [HOSTING.md §3](HOSTING.md) if images don't render.

### Step 4 — get the results

Export the project to JSON from Label Studio (**Export → JSON**), then synthesize:

```bash
uv run --extra training \
  python -m mermaid_classifier.model_review.cli synthesize \
  --tasks /tmp/review_tasks.json --export /tmp/review_export.json \
  --points-out review_points.csv --summary-out review_summary.json --notes-out review_notes.csv
```

- `review_summary.json` — the three top-level agreement rates: `v1_vs_gt`, `expert_vs_gt`,
  `expert_vs_expert`.
- `review_points.csv` — one row per labelled point, **including the image set name, source,
  and the exact `s3://` image path** so every annotation is traceable to its image.
- `review_notes.csv` — reviewers' per-image notes.

(`--tasks` is the `review_tasks.json` produced during the build. If you used `create-project`
directly you can regenerate the same file with the `build-tasks` subcommand — see below.)

---

## How the app works

<details>
<summary>The Label Studio project model, the CLI, and the modules (click to expand)</summary>

### The project a reviewer sees
Each task is one image with a fixed set of points. Every task ships three prediction layers:
- **Unlabelled Starting Set** — the blind starting layer copied into each reviewer's
  annotation so they label from scratch (this is set as the project's `model_version`).
- **ground-truth** — the CoralNet ground-truth label per point (read-only reference).
- **v1** — the reviewed model's prediction per point (read-only reference).

Points are colored by top-level category. The labeling config's taxonomy is restricted to
the reviewed model's label set (one path per class), and ground truth is rolled into that
same label set so the three layers are directly comparable.

### The CLI (`python -m mermaid_classifier.model_review.cli ...`)
- **`create-project`** — the one-command path above: build tasks from `image_set.py` →
  create a new project → attach S3 storage → import → set the blind starting layer.
- **`build-tasks`** — just builds `review_tasks.json` + `review_config.xml` from
  `image_set.py` without touching Label Studio (defaults all come from `IMAGE_SET`; every
  field is overridable with a flag). Useful for importing by hand, or to regenerate the
  tasks file needed by `synthesize`.
- **`synthesize`** — parse a Label Studio JSON export into the comparison outputs above.

### The modules
| Module | Responsibility |
| --- | --- |
| `image_set.py` | The single editable image-set definition (`ImageSet` + `IMAGE_SET`). |
| `cli.py` | Command-line entry points and the build/orchestration glue. |
| `sample.py` | Selects held-out CoralNet images and their full ground-truth points. |
| `v1_infer.py` | Runs the reviewed model over the pre-extracted feature vectors. |
| `ls_config.py` | Generates the Label Studio labeling config (taxonomy). |
| `ls_tasks.py` | Builds the tasks (points + GT/model reference layers + provenance). |
| `ls_client.py` | Thin stdlib-`urllib` Label Studio REST client. |
| `ls_export.py` | Parses a Label Studio export back into per-point expert labels. |
| `synthesis.py` | Computes the expert / ground-truth / model agreement tables. |
| `rollup.py` | Maps fine labels to top-level categories for comparison. |
| `seed.py` | (Optional) pre-seeds a blank annotation per reviewer. |

</details>

---

## Operations & infrastructure

> Only needed to **stand the app up, connect to an internally-hosted instance, back it up, or
> tear it down** — not for the everyday image-set task above.

The full operations runbook lives in **[HOSTING.md](HOSTING.md)**. It covers, step by step:

- **[§0–§1](HOSTING.md)** Prerequisites and deploying the ephemeral CloudFormation stack.
- **[§2](HOSTING.md)** Connecting via the SSM port-forward tunnel (the internal-only access path).
- **[§3](HOSTING.md)** Image serving (S3 presign mode) and the image-bucket CORS rule.
- **[§6](HOSTING.md)** Synthesizing results (also summarized above).
- **[§7](HOSTING.md)** Durability — the automatic off-instance SQLite→S3 backup and what survives what.
- **[§8](HOSTING.md)** Teardown in one step, with verification.
- **[§9](HOSTING.md)** Remote access for external reviewers (how the hosted domain is fronted).
