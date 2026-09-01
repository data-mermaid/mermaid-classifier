# Model-review Label Studio — run it (and tear it down) another day

Everything runs in **one ephemeral CloudFormation stack** (`ephemeral_ls_stack.yaml`),
reached only through an **SSM port-forward** to your own `localhost` — no public
endpoint, DNS, TLS, or key pair. Teardown is one `delete-stack` and is verifiable.

Run all commands from the `mermaid-classifier` repo root unless noted. AWS access is
the `wcs-admin` SSO profile. *(Agent: run stack + S3 API calls via the `aws-mcp`
server; the operator runs the interactive SSM tunnel and — if used — `aws sso login`.)*

## 0. Prerequisites (once)

```bash
aws sso login --profile wcs-admin              # refresh when the session expires (~1h/session)
brew install --cask session-manager-plugin     # needed for the SSM tunnel
export AWS_PROFILE=wcs-admin AWS_REGION=us-east-1
```

## 1. Deploy the stack (region us-west-2)

The stack = 1 EC2 instance (Amazon Linux 2023 on a **pinned** AMI, Docker + Label
Studio 1.13.1 + aws-cli + a 10-min sqlite→S3 backup timer, root EBS
`DeleteOnTermination`), 1 security group (no inbound), 1 IAM role/profile (SSM +
scoped read on the image prefix + read/write on the backup prefix). Nothing else is
created **by the stack**. One external prerequisite exists out of band: the
versioned backup bucket `model-review-ls-backups-554812291621` (see §7), which is
deliberately not part of the stack so backups survive teardown.

```bash
aws cloudformation create-stack --stack-name model-review-ls --region us-west-2 \
  --template-body file://mermaid_classifier/model_review/ephemeral_ls_stack.yaml \
  --capabilities CAPABILITY_IAM \
  --tags Key=Project,Value=model-review Key=Ephemeral,Value=true

aws cloudformation wait stack-create-complete --stack-name model-review-ls --region us-west-2
aws cloudformation describe-stacks --stack-name model-review-ls --region us-west-2 \
  --query "Stacks[0].Outputs"    # -> InstanceId, PortForwardCommand
```
> **Applying template changes.** UserData runs only at **launch**, so `update-stack`
> will not re-run boot changes on a live instance — recreate the stack (safe when the
> instance is empty; the backup/restore in §7 protects data otherwise). The operator's
> real `aws` CLI accepts `--template-body file://…`; the **agent's `aws-mcp` has no
> local-file access**, so it must first upload the template to a scratch S3 bucket and
> use `--template-url` (delete the scratch bucket after).

Wait ~1–2 min after `CREATE_COMPLETE` for the container to pull. Check it's healthy:
```bash
aws ssm send-command --region us-west-2 --instance-ids <InstanceId> \
  --document-name AWS-RunShellScript --parameters commands="docker ps"
# then: aws ssm get-command-invocation --command-id <id> --instance-id <InstanceId> --region us-west-2
# expect the `ls` container "Up ..." (it chowns /opt/ls_data to uid 1001 on boot)
```

## 2. Connect (SSM tunnel — run locally, leave open)

```bash
aws ssm start-session --target <InstanceId> --region us-west-2 \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["8080"],"localPortNumber":["8080"]}'
```
Open **http://localhost:8080**, create the admin account (Community edition: any
account sees all projects). Get an API token at **http://localhost:8080/api/current-user/token**
for the scripted steps below.

## 3. Images — durable S3 serving (presign mode) + CORS

Tasks carry plain **`s3://…`** image URIs (never expire). Label Studio resolves them
via an **S3 *source storage* in presign mode** (`presign=true`, `use_blob_urls=false`,
step 5): per view, LS mints a short-lived presigned S3 URL using the instance's IAM
role and 303-redirects the browser to fetch the image **straight from S3**.
Nothing is baked into the tasks and nothing expires from the reviewer's side (LS
re-mints per view).

A mixed CoralNet + MERMAID set draws from **two buckets**, so the project carries **one
source storage per bucket** and the instance role carries one scoped read grant per bucket
(`read-coralnet-images` on `coralnet-public-images/`, `read-mermaid-images` on
`coral-reef-training/mermaid/` — both in the template). LS matches each task's `s3://` URI
to the storage that covers it. Adding the MERMAID grant is an `update-stack`: it changes only
an inline IAM policy, so the instance is **not** replaced and the grant applies immediately.
Confirm that with a change set first (every `Replacement` must be `False`, and no row for
the EC2 instance).

> We tried **proxy mode** (`presign=false`, no CORS) first, but LS materializes every
> image server-side and the 50-image data manager **OOM-killed LS even on t3.medium**.
> Presign keeps LS light by offloading the fetch to the browser — the tradeoff is that
> the browser now fetches cross-origin and LS draws on a canvas, so the bucket needs a
> scoped CORS rule for the tunnel origin.

Because the browser reads the image onto a canvas cross-origin, the image bucket needs a
GET CORS rule (CORS grants no access — the object still needs the presigned signature).

**`coral-reef-training` needs nothing:** it already carries a permissive
`AllowedOrigins: ["*"]` GET/HEAD rule that other consumers depend on. **Never run
`put-bucket-cors` on it** — that call *replaces* the entire configuration.

For `dev-datamermaid-sm-sources`, add the scoped rule while reviewing and **remove it when
done**:

```bash
# ON:
aws s3api put-bucket-cors --bucket dev-datamermaid-sm-sources --cors-configuration '{
  "CORSRules":[{"AllowedMethods":["GET","HEAD"],"AllowedOrigins":["http://localhost:8080"],"AllowedHeaders":["*"],"MaxAgeSeconds":3000}]}'
# OFF (undo — restores the original no-CORS state):
aws s3api delete-bucket-cors --bucket dev-datamermaid-sm-sources
```

## 4. Build the tasks (full ground truth, in V1's label set)

Produces `<tasks>.json` (per-image points: GT + V1 predictions, colored by top-level
category) and `<config>.xml` (labeling config: colored top-level `KeyPointLabels` +
Taxonomy restricted to V1's classes + a notes box). Uses each site's manifest for the
full GT grid, mapped into V1's label set. ~70s for 50 images. It prints the realized
per-site image and point counts — check them against the weights in `image_set.py`.

**Edit `mermaid_classifier/model_review/image_set.py`** — the single place that defines
which images (and which model) a review uses. Change `name` (each distinct name makes a new
project) and any sampling knob, then build:

```bash
uv run --extra training --with pillow \
  python -m mermaid_classifier.model_review.cli build-tasks \
  --tasks-out /tmp/review_tasks.json --config-out /tmp/review_config.xml
```
All inputs default from `IMAGE_SET`. The sampling knobs are overridable with flags
(`--n-images`, `--seed`, `--min-points`, `--classifier`, …); the per-site S3 layout is
**not** — it lives only in `image_set.py`. The reviewed model defaults to
`s3://mermaid-config/classifier/v2/` and is **downloaded automatically** — no manual copy.
`build-tasks` needs valid AWS creds (`aws sso login --profile wcs-admin`) to read the
manifest, image sizes, and the model. Image URLs in the tasks are durable `s3://…` URIs.

## 5. Create the project + import (UI or API)

**One command (recommended).** With the SSM tunnel up (§2) and a token from §2, this builds
the tasks from `image_set.py` and creates a **brand-new** project (S3 presign storage, tasks
imported, blind starting layer set). It **refuses** if a project with the same `name` already
exists, so existing projects are never touched — change `name` in `image_set.py` for each set:

```bash
uv run --extra training --with pillow \
  python -m mermaid_classifier.model_review.cli create-project \
  --token <token>          # or: export LABEL_STUDIO_TOKEN=<token>
```
Then make sure the `dev-datamermaid-sm-sources` CORS rule (§3) is on so reviewers can see
the CoralNet images; `coral-reef-training` needs nothing. The manual UI/heredoc steps below
remain as a fallback.

**UI:** create a project → Settings → Labeling Interface → paste `review_config.xml`;
set Annotations-per-task minimum = number of experts; Import `review_tasks.json`.

**API (token from step 2):**
```bash
python - <<'PY'
import json, urllib.request
TOKEN="<token>"; LS="http://localhost:8080"
def post(p,b):
    r=urllib.request.Request(f"{LS}{p}",data=json.dumps(b).encode(),
        headers={"Authorization":f"Token {TOKEN}","Content-Type":"application/json"},method="POST")
    return json.loads(urllib.request.urlopen(r,timeout=120).read())
pid=post("/api/projects",{"title":"Model Review","label_config":open("/tmp/review_config.xml").read()})["id"]
# One S3 source storage per image bucket, in PRESIGN mode: LS mints a presigned S3 URL
# per view (via the instance role) and redirects the browser to fetch from S3 (LS stays
# light; needs the CORS rule from step 3). Do NOT sync them — they only resolve the
# s3:// links in the tasks. (presign=False = proxy mode, no CORS but OOMs on the 50-img
# grid.) Titles must differ so the two storages are distinguishable in the UI.
post("/api/storages/s3",{"project":pid,"bucket":"dev-datamermaid-sm-sources",
    "prefix":"coralnet-public-images/","region_name":"us-east-1",
    "use_blob_urls":False,"presign":True,"title":"coralnet-images"})
post("/api/storages/s3",{"project":pid,"bucket":"coral-reef-training",
    "prefix":"mermaid/","region_name":"us-east-1",
    "use_blob_urls":False,"presign":True,"title":"mermaid-images"})
post(f"/api/projects/{pid}/import", json.load(open("/tmp/review_tasks.json")))
# Reviewers start BLIND: pre-fill their annotation from the unlabelled starting
# layer, not from V1. Its model_version is BLANK_MODEL_VERSION ("Unlabelled
# Starting Set"). (Each task ships that + Beta + ground-truth + v1 + Comparison.)
from mermaid_classifier.model_review.ls_tasks import BLANK_MODEL_VERSION
patch=urllib.request.Request(f"{LS}/api/projects/{pid}",data=json.dumps({"model_version":BLANK_MODEL_VERSION}).encode(),
    headers={"Authorization":f"Token {TOKEN}","Content-Type":"application/json"},method="PATCH")
urllib.request.urlopen(patch,timeout=30)
print("project", pid)
PY
```
**UI equivalent of the model_version step:** Settings → Annotation (or Predictions) →
set the displayed model version to **"Unlabelled Starting Set"** so annotators start
from the unlabelled layer. Without this, they'd start pre-filled from `v1`.

**Accounts / attribution:** each reviewer must sign up with **their own account**
(their email) — LS stamps every annotation with `completed_by` (that account), which
is how §6 attributes and lets you filter by reviewer. Sharing one login makes the
work indistinguishable. In Community edition anyone who can reach LS and sign up can
see all projects, so gate *access* at the front door (see §9), not by hiding the URL.

Experts open each image to **unlabelled (grey) fixed points** and label the fine BA::GF
via the Taxonomy tree; they can toggle the `v1` / `ground-truth` / `Beta` tabs to view
references (colored by top-level category) after their pass, or `Comparison` to read all
three for one point at once; use the per-image notes box.
Optional: pre-seed one blank annotation per (task × expert) with
`mermaid_classifier.model_review.seed.seed_all(client, project_id, expert_user_ids)`.

## 6. Synthesize results

Export the project to `review_export.json` (UI Export → JSON, or `/api/projects/<id>/export?exportType=JSON`).
Also dump the accounts so annotations are attributed to reviewer **email** (each
annotation records its author as `completed_by`; `--users-json` maps id→email so
`review_points.csv`/notes carry the email and you can filter by reviewer):
```bash
curl -s -H "Authorization: Token <token>" http://localhost:8080/api/users -o review_users.json

uv run --extra training --with label-studio-sdk --with pillow \
  python -m mermaid_classifier.model_review.cli synthesize \
  --tasks /tmp/review_tasks.json --export /tmp/review_export.json --users-json review_users.json \
  --points-out review_points.csv --summary-out review_summary.json --notes-out review_notes.csv
```
Emits the three comparisons at top-level: `v1_vs_gt` (over ALL points),
`expert_vs_gt`, `expert_vs_expert`, plus a per-point table (with the reviewer's
email in the `expert` column) and notes. **Attribution requires each reviewer to
use their own account** — see §5.

## 7. Durability of expert labels — automatic S3 backup

Annotations live in **SQLite** at `/opt/ls_data/label_studio.sqlite3` on the
instance's root EBS volume. That volume is `DeleteOnTermination=true`, so the
instance is **not** the system of record — an **automatic off-instance backup** is:

- A systemd timer (`ls-backup.timer`, default every **10 min**) runs a consistent
  `sqlite3 .backup` and uploads it to
  `s3://model-review-ls-backups-554812291621/sqlite/label_studio.sqlite3`
  (external, **versioned**, private bucket — created once, out of band, and it
  **survives `delete-stack`** on purpose). A best-effort backup also runs on
  graceful shutdown (`ls-backup-shutdown.service`).
- On boot, UserData **restores** that object to `/opt/ls_data` *before* LS starts
  (only if no DB is present), so a freshly-launched instance comes back with its data.

| Event | Labels survive? |
|---|---|
| Container restart / crash (`--restart unless-stopped`) | ✅ yes (EBS) |
| Instance **reboot** / **stop→start** | ✅ yes (EBS; re-run the SSM tunnel, step 2) |
| Instance **termination / replacement** (incl. a future `create-stack`) | ✅ **restored on next boot from the last backup** — up to one timer interval (~10 min) of work can be lost |
| `delete-stack` **and** you never relaunch | ⚠️ the backup object still exists in the external bucket; restore by launching a new stack (UserData pulls it) or importing the LS JSON export |

> **The AMI is pinned** (`Parameters.Ami`) precisely so an `update-stack` cannot
> silently re-resolve "latest AL2023", replace the instance, and rely on the backup.
> Note UserData only runs at **launch** — changing it via `update-stack` does NOT
> re-run it on a live instance; recreate the stack (instance empty) to apply boot
> changes. **History:** an early `update-stack` with an SSM-resolved AMI replaced the
> instance and lost the pre-pin annotations; pinning + this backup exist to prevent a
> repeat.

Belt-and-suspenders: a JSON export is still the most portable backup, especially
before teardown:
```bash
curl -s -H "Authorization: Token <token>" \
  "http://localhost:8080/api/projects/<id>/export?exportType=JSON" -o review_export_$(date +%Y%m%d_%H%M).json
```

## 8. Teardown (one step) + verification

```bash
# 1) EXPORT FIRST — termination destroys the DB (see step 7):
curl -s -H "Authorization: Token <token>" \
  "http://localhost:8080/api/projects/<id>/export?exportType=JSON" -o review_export_final.json

# 2) Delete + verify:
aws cloudformation delete-stack --stack-name model-review-ls --region us-west-2
aws cloudformation wait stack-delete-complete --stack-name model-review-ls --region us-west-2
aws cloudformation describe-stacks --stack-name model-review-ls --region us-west-2   # should ERROR "does not exist"
aws resourcegroupstaggingapi get-resources --region us-west-2 \
  --tag-filters Key=Project,Values=model-review --query "ResourceTagMappingList[].ResourceARN"   # must be []

# 3) Remove the image-bucket CORS rule added in step 3:
aws s3api delete-bucket-cors --bucket dev-datamermaid-sm-sources
```
The LS data volume dies with the instance; the image-bucket read grant and all other
resources go with the stack. **The external backup bucket
(`model-review-ls-backups-554812291621`) is intentionally NOT part of the stack and
survives** — that is where your last sqlite backup lives. Delete it manually only when
you are certain you no longer need the annotations:
```bash
aws s3 rm s3://model-review-ls-backups-554812291621 --recursive --region us-west-2
aws s3api delete-bucket --bucket model-review-ls-backups-554812291621 --region us-west-2
```

## 9. Remote access for external reviewers (when you outgrow the SSM tunnel)

The SSM tunnel (§2) is internal-only and fine for one operator, but external experts
can't easily run it. When you expose LS to reviewers, **don't rely on an obscure
domain** — LS Community shows every project to any account that can reach it and sign
up, so a guessable/leaked/crawled URL = open data. Gate access at the front door by
*identity*, cheapest first:

- **Identity-gated tunnel (recommended, ~free, no public port).** Put the instance on
  **Tailscale** (reviewers join the tailnet) or front LS with **Cloudflare Tunnel +
  Cloudflare Access** (allowlist reviewer emails / Google login). No inbound SG rule,
  no ALB, and you revoke per-person. Reviewers then self-register their LS account
  behind the gate.
- **Public URL done properly.** ALB/CloudFront + Route53 + ACM TLS, plus an SG/WAF IP
  allowlist. Real, but the most billable/standing infra (what we deliberately avoided).

Whichever front door you pick, also set **`LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true`**
on the container so LS signup requires the org invite link (defense in depth), and keep
self-registration on so each reviewer makes their own attributable account (§5).
