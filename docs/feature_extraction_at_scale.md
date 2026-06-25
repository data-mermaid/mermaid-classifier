# Feature extraction at scale (SageMaker Processing Jobs)

`scripts/build_feature_bucket.py` is GPU-bound and single-process: it
walks 192 CoralNet sources sequentially, which takes days on a single
laptop. To finish in a few hours instead, run it in parallel as
`scripts/launch_feature_extraction_sagemaker.py`, which fans out N
SageMaker Processing Jobs -- each handling a sharded subset of source
IDs against its own GPU instance.

Between runs only two AWS resources persist: one ECR repository and one
IAM role (both in the compute account). Every Processing Job spins up
its own GPU instance, runs the container, and self-terminates -- there
is nothing to tear down after a backfill.

## Account topology

The MERMAID setup spans two AWS accounts. The split below is what this
runbook assumes; only S3 lives in the storage account, everything else
(compute, container registry, IAM) lives in the data-science account.

| Account              | ID                       | Holds                                                                 |
| -------------------- | ------------------------ | --------------------------------------------------------------------- |
| Data-science compute | `<compute-account-id>`   | ECR repo (worker image), IAM role, SageMaker Processing Jobs          |
| MERMAID storage      | `<storage-account-id>`   | Source bucket `<source-image-bucket>` and the feature target bucket (e.g. `<target-feature-bucket>`) |

The cross-account piece is straightforward: the IAM role in the compute
account is granted read on the source bucket and read/write on the
target bucket via two bucket policies in the storage account. Those
bucket policies are a one-time admin action; everything else is
self-service.

The `efficientnet_weights.pt` file lives inside the target bucket
(e.g. `s3://<target-feature-bucket>/efficientnet_weights.pt`)
so the same bucket policy covers both reads and writes -- no separate
weights bucket is needed.

## One-time setup

Do these once. Steps 1-3 happen in the compute account
(`<compute-account-id>`); step 4 happens in the storage account
(`<storage-account-id>`).

### 1. Build + push the worker image to ECR

Sign in to the compute account with the AdministratorAccess role:

```bash
aws sso login --profile <compute-account-admin-profile>
export AWS_PROFILE=<compute-account-admin-profile>
aws sts get-caller-identity   # confirm you are in the compute account
```

Create the ECR repo (idempotent -- safe to re-run):

```bash
aws ecr create-repository \
    --repository-name mermaid-features \
    --region us-east-1
```

Build the image (the base is the official PyTorch image on Docker Hub
-- public, no ECR auth needed for the build itself) and push:

```bash
cd mermaid-classifier
ACCT=<compute-account-id>
IMAGE="${ACCT}.dkr.ecr.us-east-1.amazonaws.com/mermaid-features:latest"

aws ecr get-login-password --region us-east-1 \
    | docker login --username AWS --password-stdin "${ACCT}.dkr.ecr.us-east-1.amazonaws.com"

docker buildx build \
    --platform linux/amd64 \
    -t "${IMAGE}" \
    -f docker/feature_extraction/Dockerfile \
    .

docker push "${IMAGE}"
```

Expected image size: ~6-7 GB (PyTorch base is ~5 GB; project + pyspacer
deps add ~1-2 GB). The first push uploads everything; subsequent pushes
ship only changed layers (typically just the `pip install` layer when
project deps change).

Smoke-check before pushing -- it should print `build_feature_bucket.py`'s
help:

```bash
docker run --rm "${IMAGE}" --help | head
```

### 2. Upload the weights file into the target bucket

The weights live inside the target bucket so there is only one bucket
policy to manage on the storage side. Use whatever profile you already
use to read/write the MERMAID storage account (often `wcs`):

```bash
aws s3 cp /path/to/efficientnet_weights.pt \
    s3://<target-feature-bucket>/efficientnet_weights.pt \
    --profile wcs
```

If you don't have write access to that bucket from any local profile,
ask a storage-account admin to upload the file. It only needs to land
there once.

### 3. Create the IAM role in the compute account

Save the trust policy as `trust.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "sagemaker.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
```

Save the identity policy as `policy.json` (substitute the real target
bucket name if it differs):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadSourceBucketCrossAccount",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::<source-image-bucket>",
        "arn:aws:s3:::<source-image-bucket>/coralnet-public-images/*"
      ]
    },
    {
      "Sid": "ReadWriteTargetBucketCrossAccount",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:PutObjectAcl",
        "s3:ListBucket",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::<target-feature-bucket>",
        "arn:aws:s3:::<target-feature-bucket>/*"
      ]
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:us-east-1:<compute-account-id>:log-group:/aws/sagemaker/ProcessingJobs*"
    },
    {
      "Sid": "ECRPull",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer"
      ],
      "Resource": "*"
    }
  ]
}
```

Create the role and attach the policy:

```bash
aws iam create-role \
    --role-name MermaidFeatureExtractionRole \
    --assume-role-policy-document file://trust.json

aws iam put-role-policy \
    --role-name MermaidFeatureExtractionRole \
    --policy-name MermaidFeatureExtractionPolicy \
    --policy-document file://policy.json

aws iam get-role --role-name MermaidFeatureExtractionRole --query 'Role.Arn'
```

Save the printed ARN -- you will pass it as `--role-arn` to the
launcher.

### 4. Storage-account admin: apply two cross-account bucket policies

This is the only step that requires admin access to the storage account
(`<storage-account-id>`). The admin should **merge** these statements into
each bucket's existing policy (do not replace; the buckets likely
already have policies serving other consumers).

On the source bucket `<source-image-bucket>`, add:

```json
{
  "Sid": "AllowMermaidFeatureExtractionCrossAccountRead",
  "Effect": "Allow",
  "Principal": {
    "AWS": "arn:aws:iam::<compute-account-id>:role/MermaidFeatureExtractionRole"
  },
  "Action": ["s3:GetObject", "s3:ListBucket"],
  "Resource": [
    "arn:aws:s3:::<source-image-bucket>",
    "arn:aws:s3:::<source-image-bucket>/coralnet-public-images/*"
  ]
}
```

On the target bucket (e.g. `<target-feature-bucket>`), add:

```json
{
  "Sid": "AllowMermaidFeatureExtractionCrossAccountReadWrite",
  "Effect": "Allow",
  "Principal": {
    "AWS": "arn:aws:iam::<compute-account-id>:role/MermaidFeatureExtractionRole"
  },
  "Action": [
    "s3:GetObject",
    "s3:PutObject",
    "s3:PutObjectAcl",
    "s3:ListBucket",
    "s3:DeleteObject"
  ],
  "Resource": [
    "arn:aws:s3:::<target-feature-bucket>",
    "arn:aws:s3:::<target-feature-bucket>/*"
  ]
}
```

The admin can apply each policy via:

```bash
# Pull the existing policy first
aws s3api get-bucket-policy --bucket <bucket> \
    --query Policy --output text > current-policy.json
# Manually merge the new Statement into current-policy.json
aws s3api put-bucket-policy --bucket <bucket> \
    --policy file://current-policy.json
```

**Object ownership note for the target bucket:** when the compute-account
role writes feature vectors, the writer's account owns the resulting
objects by default. If the target bucket has the modern setting
**"Object Ownership: Bucket owner enforced"** (S3 default since
April 2023), this is solved automatically. If it's still "Object
writer", ask the admin to flip it -- otherwise downstream readers in
the storage account will need explicit grants for the writer-owned
objects.

### 5. Service quota check (optional)

If `ml.g5.xlarge` shows zero in Service Quotas > Amazon SageMaker in
the compute account, request a bump to at least the worker count you
plan to run (default 16). Use the
"`ml.g5.xlarge` for processing job usage" quota.

## Launching a run

Once the one-time setup is in place, every backfill is a single
command. From the compute account:

```bash
cd mermaid-classifier
export AWS_PROFILE=<compute-account-admin-profile>

uv run python scripts/launch_feature_extraction_sagemaker.py \
    --sources-csv ../sources/CoralNetSourcesFirst192.csv \
    --target-bucket <target-feature-bucket> \
    --weights-s3-uri s3://<target-feature-bucket>/efficientnet_weights.pt \
    --ecr-image <compute-account-id>.dkr.ecr.us-east-1.amazonaws.com/mermaid-features:latest \
    --role-arn arn:aws:iam::<compute-account-id>:role/MermaidFeatureExtractionRole \
    --workers 16
```

The launcher then:

1. Reads the source-ID CSV.
2. Round-robin-splits the IDs into 16 equal chunks (~12 sources each).
3. Submits 16 SageMaker Processing Jobs in parallel; each runs the
   container against its chunk on its own `ml.g5.xlarge`.
4. Polls every 60s and prints a summary on completion (Completed /
   Failed / Stopped counts, plus per-job names).

Per-job stdout/stderr stream to CloudWatch under
`/aws/sagemaker/ProcessingJobs/<job-name>`. Tail one to confirm
throughput:

```bash
aws logs tail \
    /aws/sagemaker/ProcessingJobs \
    --follow \
    --filter-pattern "mermaid-features-<run-id>-0"
```

## Dry run

Adding `--dry-run` prints the planned chunking and estimated cost
without submitting anything:

```bash
uv run python scripts/launch_feature_extraction_sagemaker.py \
    --sources-csv ../sources/CoralNetSourcesFirst192.csv \
    --target-bucket <target-feature-bucket> \
    --weights-s3-uri s3://<target-feature-bucket>/efficientnet_weights.pt \
    --ecr-image <compute-account-id>.dkr.ecr.us-east-1.amazonaws.com/mermaid-features:latest \
    --role-arn arn:aws:iam::<compute-account-id>:role/MermaidFeatureExtractionRole \
    --workers 16 \
    --dry-run
```

A full 192-source run on 16x `ml.g5.xlarge` runs ~$60-70 (~3h per
worker). Processing Jobs do not support spot pricing, but the cost is
small enough that the on-demand premium isn't worth migrating to AWS
Batch for.

## Smoke test before a real backfill

Run one worker against one small source to validate the image, role,
and bucket policies end-to-end:

```bash
uv run python scripts/launch_feature_extraction_sagemaker.py \
    --source-ids 1968 \
    --target-bucket <target-feature-bucket> \
    --weights-s3-uri s3://<target-feature-bucket>/efficientnet_weights.pt \
    --ecr-image <compute-account-id>.dkr.ecr.us-east-1.amazonaws.com/mermaid-features:latest \
    --role-arn arn:aws:iam::<compute-account-id>:role/MermaidFeatureExtractionRole \
    --workers 1 \
    --instance-type ml.g4dn.xlarge
```

`ml.g4dn.xlarge` is cheaper than `ml.g5.xlarge` and adequate for a
smoke test. Watch CloudWatch for per-image progress lines, then
spot-check the target bucket:

```bash
aws s3 ls s3://<target-feature-bucket>/s1968/features/ \
    --profile wcs | wc -l
```

## Resuming after partial failure

Every write inside the worker is idempotent at the (source, image)
level. `--skip-existing` (always enabled by the launcher) checks the
target bucket and only extracts missing feature vectors. If any worker
fails -- transient S3 error, OOM on a giant source -- just re-run the
launcher with the same arguments. Already-completed work is skipped;
only the remainder is re-extracted.

## What lives where

| Path | Purpose |
| --- | --- |
| `scripts/launch_feature_extraction_sagemaker.py` | Local launcher; submits + polls Processing Jobs |
| `scripts/build_feature_bucket.py` | Worker script (runs inside the container) |
| `docker/feature_extraction/Dockerfile` | Worker container definition |
| `docker/feature_extraction/entrypoint.sh` | Container entrypoint (execs the worker script) |
| `.dockerignore` | Keeps the build context small |
| `tests/test_launch_feature_extraction.py` | Unit tests for the launcher (mocked AWS) |
