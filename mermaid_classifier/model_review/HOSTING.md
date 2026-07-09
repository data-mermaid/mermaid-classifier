# Ephemeral Label Studio hosting for the model-review experiment

**Internal-only.** Everything runs inside **one CloudFormation stack**
(`ephemeral_ls_stack.yaml`) with **no public endpoint** — no inbound ports, no
DNS, no TLS cert, no load balancer. You reach Label Studio through an **SSM
port-forwarding tunnel** to `http://localhost:8080` on your own machine. Teardown
is atomic and **verifiable**: `delete-stack` removes exactly what the stack made,
and a tag scan afterward proves nothing is left. Nothing is created outside the
stack (no S3 buckets, Route 53 records, ALB/ACM, key pairs, or log groups).

When the tool is proven useful, a public-facing layer (for external experts) can be
added as a *separate* opt-in step — not created now.

*(Agent: run stack create/delete/describe via the `aws-mcp` server. The interactive
SSM tunnel is run by the operator locally — see below.)*

## What the stack contains (the complete inventory)

- 1 EC2 instance (`t3.medium`, Amazon Linux 2023), root EBS `DeleteOnTermination=true`
- 1 security group with **no inbound rules** (egress only, for docker pull + SSM)
- 1 IAM role + instance profile (SSM Session Manager only — no SSH key pair)

## Deploy

```bash
aws cloudformation create-stack \
  --stack-name model-review-ls \
  --template-body <ephemeral_ls_stack.yaml> \
  --capabilities CAPABILITY_IAM \
  --tags Key=Project,Value=model-review Key=Ephemeral,Value=true

aws cloudformation wait stack-create-complete --stack-name model-review-ls
aws cloudformation describe-stacks --stack-name model-review-ls \
  --query "Stacks[0].Outputs"     # -> InstanceId, PortForwardCommand, LocalUrl
```

## Connect (SSM tunnel — run locally)

Prereq once: the AWS CLI **session-manager-plugin**
(`brew install --cask session-manager-plugin`). Then, with `wcs-admin` creds:

```bash
aws ssm start-session --target <InstanceId> \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["8080"],"localPortNumber":["8080"]}'
```

Leave that running, then open **http://localhost:8080**. Create the admin account
(and any internal reviewer accounts). Because the tunnel forwards to your own
`localhost`, the review CLI/SDK can also target `http://localhost:8080` while it is up.

## Load the project

The review CLI and the SDK seeding step are run ephemerally, not from an installed
extra: `uv run --extra training --with label-studio-sdk --with pillow python -m
mermaid_classifier.model_review.cli ...` (the seeding helper likewise needs
`--with label-studio-sdk`). This keeps `label-studio-sdk` out of the shared `uv.lock`.

1. Create a project. Paste `review_config.xml` (from `build-tasks`) into
   Settings → Labeling Interface.
2. Settings → General → set **Annotations per task minimum** = number of experts.
3. Import `review_tasks.json` (Import button, or the SDK in the seeding step).

> **Presigned-URL lifetime caveat.** `build-tasks` presigns image URLs for 7 days,
> but if you ran it under **SSO / STS temporary credentials** (`wcs-admin`), the URL
> is only valid until the **session token** expires (hours), not 7 days — images
> then 403. For a longer review, presign with long-lived IAM credentials, make
> `coralnet-public-images` reachable another way, or re-run `build-tasks` + re-import.

## Teardown (one step) + verification

```bash
# 1) Export first (synthesize needs review_export.json), THEN:
aws cloudformation delete-stack --stack-name model-review-ls
aws cloudformation wait stack-delete-complete --stack-name model-review-ls

# 2) Prove the stack is gone (this call should ERROR "does not exist"):
aws cloudformation describe-stacks --stack-name model-review-ls

# 3) Prove nothing tagged Project=model-review remains anywhere (must be empty):
aws resourcegroupstaggingapi get-resources \
  --tag-filters Key=Project,Values=model-review \
  --query "ResourceTagMappingList[].ResourceARN"
```

If step 2 errors with "Stack ... does not exist" **and** step 3 returns `[]`, teardown
is complete and verified.
