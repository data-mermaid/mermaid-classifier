# Ephemeral Label Studio hosting for the model-review experiment

Single, tagged, throwaway host. **Tear down in one step when done.**

## Deploy (small EC2 + Docker)

1. Launch one `t3.medium` Ubuntu instance, tagged `Project=model-review Ephemeral=true`,
   security group allowing inbound 443 (and 22 for you). *(Agent: use the aws-mcp server
   for these API calls, not the aws CLI.)*
2. On the instance:
   ```bash
   docker run -d --name ls -p 80:8080 \
     -v /opt/ls_data:/label-studio/data \
     -e LABEL_STUDIO_HOST=https://<your-domain-or-ip> \
     heartexlabs/label-studio:1.13.1
   ```
   Put HTTPS in front (Caddy one-liner or an ALB) so external experts get a valid cert.
3. Open the URL, create the admin account, then create each expert's account
   (Organization → People → invite / add). Community edition: everyone shares one org.

## Load the project

The review CLI and the SDK seeding step are run ephemerally, not from an
installed extra: `uv run --extra training --with label-studio-sdk --with
pillow python -m mermaid_classifier.model_review.cli ...` (the seeding helper
likewise needs `--with label-studio-sdk`). This keeps `label-studio-sdk` and
its transitive deps out of the shared `uv.lock`.

1. Create a project. Paste `review_config.xml` (from `build-tasks`) into
   Settings → Labeling Interface.
2. Settings → General → set **Annotations per task minimum** = number of experts
   (so every expert labels every image).
3. Import `review_tasks.json` (Import button, or the SDK in the seeding step).

## Teardown (one step)

Export first (the `synthesize` step needs the export), then:
```bash
# Agent: via aws-mcp — terminate the single tagged instance.
# Nothing else persists; presigned URLs expire on their own.
```
Confirm no instance with tag `Project=model-review` remains. There is no CDK stack,
load balancer left running, or bucket to clean beyond the ephemeral instance.
