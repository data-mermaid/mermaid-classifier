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

The stack = 1 EC2 instance (Amazon Linux 2023, Docker + Label Studio 1.13.1, root
EBS `DeleteOnTermination`), 1 security group (no inbound), 1 IAM role/profile (SSM
only). Nothing else is created.

```bash
aws cloudformation create-stack --stack-name model-review-ls --region us-west-2 \
  --template-body file://mermaid_classifier/model_review/ephemeral_ls_stack.yaml \
  --capabilities CAPABILITY_IAM \
  --tags Key=Project,Value=model-review Key=Ephemeral,Value=true

aws cloudformation wait stack-create-complete --stack-name model-review-ls --region us-west-2
aws cloudformation describe-stacks --stack-name model-review-ls --region us-west-2 \
  --query "Stacks[0].Outputs"    # -> InstanceId, PortForwardCommand
```
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

## 3. Image CORS (required, and REMEMBER to undo)

Label Studio draws each image on a canvas, so the private image bucket must return
CORS headers for the tunnel origin. The bucket has **no CORS by default**; add a
scoped GET rule while reviewing, and **remove it when done**. (CORS grants no new
access — objects still need the presigned signature.)

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
Taxonomy restricted to V1's classes + a notes box). Uses the CoralNet manifest for
the full GT grid, mapped into V1's label set. ~70s for 50 images.

```bash
uv run --extra training --with label-studio-sdk --with pillow \
  python -m mermaid_classifier.model_review.cli build-tasks \
  --n-images 50 --min-points 15 --seed 1 \
  --classifier /Users/gregn/Documents/wcs/models/v1 \
  --v1-rollup-csv sagemaker/configs/coralnet_top108_full/rollups.csv \
  --heldout-csv /Users/gregn/Documents/wcs/reports/model_benchmark/data/v1_annotations_val.csv \
  --tasks-out /tmp/review_tasks.json --config-out /tmp/review_config.xml
```
Key flags: `--manifest-uri` (default = the top108_full CoralNet manifest parquet),
`--feature-bucket` (default `2605-coralnet-public-sources`), `--image-bucket` /
`--image-key-template` (default the CoralNet display images in
`dev-datamermaid-sm-sources/coralnet-public-images/...`).

> **Presigned-URL lifetime caveat.** Image URLs are presigned for 7 days, but under
> SSO/STS temporary creds they only work until the session token expires (hours).
> For a multi-day review, presign with long-lived IAM creds, or re-run `build-tasks`
> + re-import before each session. (403s on images = expired URLs; re-run build-tasks.)

## 5. Create the project + import (UI or API)

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
post(f"/api/projects/{pid}/import", json.load(open("/tmp/review_tasks.json")))
# Reviewers start BLIND: pre-fill their annotation from the unlabelled starting
# layer, not from V1. Its model_version is BLANK_MODEL_VERSION ("Unlabelled
# Starting Set"). (Each task ships that + ground-truth + v1 predictions.)
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

Experts open each image to **unlabelled (grey) fixed points** and label the fine BA::GF
via the Taxonomy tree; they can toggle the `ground-truth` / `v1` tabs to view references
(colored by top-level category) after their pass; use the per-image notes box.
Optional: pre-seed one blank annotation per (task × expert) with
`mermaid_classifier.model_review.seed.seed_all(client, project_id, expert_user_ids)`.

## 6. Synthesize results

Export the project to `review_export.json` (UI Export → JSON, or `/api/projects/<id>/export?exportType=JSON`), then:
```bash
uv run --extra training --with label-studio-sdk --with pillow \
  python -m mermaid_classifier.model_review.cli synthesize \
  --tasks /tmp/review_tasks.json --export /tmp/review_export.json \
  --points-out review_points.csv --summary-out review_summary.json --notes-out review_notes.csv
```
Emits the three comparisons at top-level: `v1_vs_gt` (over ALL points),
`expert_vs_gt`, `expert_vs_expert`, plus per-point table and notes.

## 7. Durability of expert labels — READ THIS

Annotations are stored in **SQLite** at `/opt/ls_data/label_studio.sqlite3`, on the
instance's **root EBS volume** (bind-mounted). There is **no automatic off-instance
backup** in this ephemeral design.

| Event | Labels survive? |
|---|---|
| Container restart / crash (`--restart unless-stopped`) | ✅ yes |
| Instance **reboot** (OS restart) | ✅ yes — Docker auto-starts, EBS persists (just re-run the SSM tunnel, step 2) |
| Instance **stop → start** | ✅ yes — EBS persists; SSM tunnel still works (uses instance id, not IP) |
| Instance **termination** (incl. `delete-stack`, spot/hardware) | ❌ **LOST** — root volume is `DeleteOnTermination=true`, no backup |

**Therefore: export the annotations regularly, and ALWAYS before teardown.** The LS
JSON export is the canonical, restore-independent backup:
```bash
# through the tunnel; token from step 2
curl -s -H "Authorization: Token <token>" \
  "http://localhost:8080/api/projects/<id>/export?exportType=JSON" -o review_export_$(date +%Y%m%d_%H%M).json
```
Keep those files somewhere durable (your machine / S3). Reboots and stop/starts are
safe, so you do NOT need to export just to restart — only to guard against
termination. (If you later want automatic durability, add a scoped S3-write policy to
the instance role + a cron that copies the sqlite to S3, or point LS at RDS Postgres.)

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
Presigned URLs expire on their own; the LS data volume dies with the instance.
