# Model-review annotation app

A tool for evaluating a MERMAID classifier against expert annotations. It samples
held-out images from both CoralNet and MERMAID, lays down each image's ground-truth points, and hosts them in
[**Label Studio**](https://model-review.datamermaid.org/) so reviewers can label each point
by hand. Their labels are then compared against both the ground truth and the model's
predictions.

**Live app: https://model-review.datamermaid.org/**

**Reviewers:** see the **[Reviewer guide](REVIEWER_GUIDE.md)** — step-by-step, with
screenshots, for how to log in and annotate.

---

# Change the image set (the common task)

> This is what you do to put a **new set of images** in front of reviewers. It creates a
> **brand-new** Label Studio project and never touches existing ones. The app is already
> deployed — you only open a short-lived **SSM tunnel** to reach it; you never deploy or tear
> anything down.

### What you need first

(Requires the AWS CLI + Session Manager plugin — see [HOSTING.md §0](HOSTING.md).)

1. **AWS access** (your own profile) — reads images/manifest/model from S3 *and* opens the tunnel:
   ```bash
   aws sso login --profile wcs-admin
   export AWS_PROFILE=wcs-admin
   ```
2. **An SSM tunnel to the app.** The script talks to Label Studio's API over `localhost:8080`;
   the public URL is browser-only (it sits behind Cloudflare Access). Open the tunnel in a
   **separate terminal** and leave it running:
   ```bash
   IID=$(aws cloudformation describe-stacks --stack-name model-review-ls --region us-west-2 \
     --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" --output text)
   aws ssm start-session --target "$IID" --region us-west-2 \
     --document-name AWS-StartPortForwardingSession \
     --parameters '{"portNumber":["8080"],"localPortNumber":["8080"]}'
   ```
   Confirm it's really Label Studio (not some other local server on 8080):
   ```bash
   curl -s localhost:8080/api/version/     # expect JSON: {"release": "1.13.1", ...}
   ```
   If that 404s, another process owns 8080 — rerun the tunnel with `"localPortNumber":["9091"]`
   and use `--ls-url http://localhost:9091` in Step 2.
3. **A Label Studio API token.** In a browser, open https://model-review.datamermaid.org/,
   sign in, and copy your token from **Account & Settings → Access Token** (or
   `/api/current-user/token`):
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
| `n_images` | How many images to sample **in total**, split across the sites by their weights. |
| `min_points` | Only sample images with at least this many ground-truth points. |
| `seed` | Random seed — change it to draw a different random sample of images. |
| `heldout_csv` | The held-out validation split that defines which images are eligible. |
| `beta_classifier` | Beta pickle, scored in an isolated scikit-learn 1.1.3 subprocess. |
| `v1_rollup_csv` | Rollup that maps ground truth into the model's label set. |
| `sites` | One `SiteSpec` per data source — see below. |

Each `SiteSpec` says where one site's data lives and how big its share is:

| `SiteSpec` field | What it controls |
| --- | --- |
| `site` | `coralnet` or `mermaid` — matches the `site` column of `heldout_csv`. |
| `weight` | Relative share of `n_images`. The shipped `2.0` / `1.0` gives **13 CoralNet + 7 MERMAID** at `n_images=20`. |
| `manifest_uri` | Parquet holding this site's full ground-truth grid per image. |
| `feature_bucket` | Bucket with the per-image feature vectors (used for model inference). |
| `image_bucket` / `image_prefix` / `image_key_template` | Where this site's display images live in S3. |
| `image_key_extensions` | Extensions to probe for `{ext}` in the key template. Empty means the template is exact. MERMAID needs `("png", "jpg", "jpeg")`. |

**Sampling.** Each site is a separate stratum drawn independently: its candidates are
shuffled once with a seed derived from `(seed, site)`, then walked in order keeping images
that have enough usable ground-truth points *and* a display object in S3, until its share
is filled. That makes each share an exact random sample of its own pool, and means
changing one site's `weight` never moves another site's draw. If a site cannot fill its
share, the build fails loudly rather than quietly returning a smaller project.

The quota is on *images*, so the realized point-level split differs from the weights —
MERMAID grids are a fixed 25 points, CoralNet's average ~20. `build-tasks` prints both.

For the common case ("same as before, but a different batch of images") you only touch
`name`, `n_images`, and `seed`. Adding a site, or changing where its data lives, is the
only reason to touch `sites`.

### Step 2 — run one command

```bash
uv run --extra training --with pillow \
  python -m mermaid_classifier.model_review.cli create-project \
  --ls-url http://localhost:8080
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
benthic-attribute label to each via the taxonomy tree. They can toggle the read-only
`v1`, `ground-truth` and `Beta` reference layers *after* their pass, and use the per-image
notes box.

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

- `review_summary.json` — the four top-level agreement rates: `v1_vs_gt`, `beta_vs_gt`,
  `expert_vs_gt`, `expert_vs_expert`.
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
Each task is one image with a fixed set of points. Every task ships five prediction layers:
- **Unlabelled Starting Set** — the blind starting layer copied into each reviewer's
  annotation so they label from scratch (this is set as the project's `model_version`).
- **Beta** — what the model currently deployed to the MERMAID API predicts (read-only
  reference), so the reviewed model is judged against the incumbent, not just ground truth.
- **ground-truth** — the source dataset's ground-truth label per point (read-only reference).
- **v1** — the reviewed model's prediction per point (read-only reference).
- **Comparison** — every set's fine label for one point, in a single perRegion text block.
  Selecting a point shows ground truth, `v1` and `Beta` together, with each model marked
  match or differs against ground truth. This exists because switching tabs discards the
  selected region — each annotation is its own store in the editor, and Community edition
  has no side-by-side compare — so comparing one point across tabs is otherwise a matter of
  re-finding it after every click.

Label Studio assigns prediction ids in that array order and lists the tabs by **descending**
id, so on screen a reviewer sees: their own annotation, `Comparison`, `v1`, `ground-truth`,
`Beta`, `Unlabelled Starting Set`. The order of the list in `ls_tasks.build_task` is what
controls this; there is no per-project tab-order setting. `Comparison` is last in the array
so it sits beside the reviewer's own tab — also where a later-posted prediction would land,
since a new prediction takes the highest id.

The reviewer's own labels are **not** in the comparison block: predictions are baked into the
task at import, before anyone has annotated. Adding them needs a second pass that reads
annotations back and posts an updated prediction; `comparison.comparison_text` already takes
the sets as an ordered sequence, so that is a longer list rather than a new format.

Points are colored by top-level category. The labeling config's taxonomy is the reviewed
model's label set plus the Beta labels this image set actually shows (one path per class), and
ground truth is rolled into the reviewed model's label set so the layers are comparable.

### The CLI (`python -m mermaid_classifier.model_review.cli ...`)
- **`create-project`** — the one-command path above: build tasks from `image_set.py` →
  create a new project → attach one S3 storage per image bucket → import → set the
  blind starting layer.
- **`build-tasks`** — just builds `review_tasks.json` + `review_config.xml` from
  `image_set.py` without touching Label Studio. Defaults come from `IMAGE_SET`; the
  sampling knobs (`--name`, `--n-images`, `--seed`, `--min-points`, `--classifier`,
  `--beta-classifier`, `--heldout-csv`, `--v1-rollup-csv`) are overridable with flags, but
  the per-site S3
  layout lives only in `image_set.py`. Useful for importing by hand, or to regenerate the
  tasks file needed by `synthesize`.
- **`synthesize`** — parse a Label Studio JSON export into the comparison outputs above.

### The modules
| Module | Responsibility |
| --- | --- |
| `image_set.py` | The single editable image-set definition (`ImageSet` + `IMAGE_SET`). |
| `cli.py` | Command-line entry points and the build/orchestration glue. |
| `sample.py` | Draws the stratified sample of held-out images and their full ground-truth points. |
| `features.py` | Loads the pre-extracted feature vectors once, for every model to score. |
| `v1_infer.py` | Runs the reviewed model over those feature vectors. |
| `beta_infer.py` | Runs the Beta model out of process (its pickle needs scikit-learn 1.1.3). |
| `beta_score.py` | The scorer that runs inside that isolated environment. |
| `comparison.py` | Renders one point's label from every set into a single text block. |
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
