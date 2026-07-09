# Ephemeral Label Studio hosting for the model-review experiment

Everything runs inside **one CloudFormation stack** (`ephemeral_ls_stack.yaml`), so
teardown is atomic and **verifiable**: `delete-stack` removes exactly what the stack
created, and a tag scan afterward proves nothing is left. No S3 buckets, Route 53
records, ALB/ACM, key pairs, or CloudWatch log groups are created.

*(Agent: run all AWS calls via the `aws-mcp` server, not the `aws` CLI.)*

## What the stack contains (the complete inventory)

- 1 EC2 instance (`t3.medium`, Amazon Linux 2023), root EBS volume `DeleteOnTermination=true`
- 1 security group (inbound 80 + 443)
- 1 IAM role + instance profile (SSM Session Manager only — no SSH key pair)

TLS is served by Caddy (auto Let's Encrypt) on `<public-ip>.sslip.io`; the cert lives
on the instance and dies with it. Shell access is `aws ssm start-session` (no key pair).

## Deploy

```bash
aws cloudformation create-stack \
  --stack-name model-review-ls \
  --template-body <ephemeral_ls_stack.yaml> \
  --capabilities CAPABILITY_IAM \
  --tags Key=Project,Value=model-review Key=Ephemeral,Value=true

aws cloudformation wait stack-create-complete --stack-name model-review-ls
aws cloudformation describe-stacks --stack-name model-review-ls \
  --query "Stacks[0].Outputs"        # -> Url (https://<ip>.sslip.io), ShellCommand
```

Wait ~2-3 min after `CREATE_COMPLETE` for Caddy to obtain its cert, then open the `Url`.
Create the admin account, then each expert's account (Organization → People). Community
edition: everyone shares one org.

## Load the project

The review CLI and the SDK seeding step are run ephemerally, not from an installed
extra: `uv run --extra training --with label-studio-sdk --with pillow python -m
mermaid_classifier.model_review.cli ...` (the seeding helper likewise needs
`--with label-studio-sdk`). This keeps `label-studio-sdk` out of the shared `uv.lock`.

1. Create a project. Paste `review_config.xml` (from `build-tasks`) into
   Settings → Labeling Interface.
2. Settings → General → set **Annotations per task minimum** = number of experts
   (so every expert labels every image).
3. Import `review_tasks.json` (Import button, or the SDK in the seeding step).

> **Presigned-URL lifetime caveat.** `build-tasks` presigns image URLs for 7 days,
> but if you ran it under **SSO / STS temporary credentials** (`wcs-admin`), the URL
> is only valid until the **session token** expires (hours), not 7 days — remote
> experts will then get 403s on the images. For a multi-day review, either presign
> with long-lived IAM credentials, make `coralnet-public-images` readable to the
> reviewers another way, or re-run `build-tasks` + re-import before each session.

## Teardown (one step) + verification

```bash
# 1) Export first (the synthesize step needs review_export.json), THEN:
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
is complete and verified. Presigned URLs simply expire on their own; nothing else was
created outside the stack.
