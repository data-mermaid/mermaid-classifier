# MLflow notes

MLflow has two separate concerns that are easy to conflate:

1. **Logging** runs and models — happens automatically during training (and is
   what annotation/inference read back from). It's controlled by the
   `MLFLOW_TRACKING_SERVER` setting (the tracking URI). **No server process is
   needed to log.**
2. **Viewing** logged runs/models in a browser — *that's* what an MLflow server /
   UI is for. You only need it to look at results, not to produce them.

## Where runs get logged (the tracking URI)

Set `MLFLOW_TRACKING_SERVER` in your `.env` (see `.env.example`):

- **Local dev** — point it at a local store and MLflow writes there **directly,
  with nothing to start**. A SQLite DB is the usual choice:

      MLFLOW_TRACKING_SERVER=sqlite:///mlflow.db

  (a `file:./mlruns` directory store works too). Training and annotation log
  straight to that DB/directory.

- **SageMaker** — point it at MERMAID's **SageMaker-managed MLflow App** by its
  ARN:

      MLFLOW_TRACKING_SERVER=arn:aws:sagemaker:<region>:<account>:mlflow-app/<app-id>

  (discover the exact ARN with `aws sagemaker list-mlflow-apps` — see
  [training_at_scale.md](training_at_scale.md))

  The App is always running (provisioned by IaC — see
  [training_at_scale.md](training_at_scale.md)), so there's **nothing to start or
  wait for**: with the ARN set, runs are logged to it automatically.

## Viewing logged runs and models

This is the only part that needs a server/UI:

- **Local** — start the MLflow UI against the same store you logged to, e.g.
  `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 8080`, then open
  `http://localhost:8080`. (This is purely a viewer; it is not required in order
  to log.)
- **SageMaker** — in SageMaker Studio, open the MLflow App (Studio → MLflow →
  the App → Open MLflow).

For a self-contained HTML report of a single run — no MLflow UI required — use
`scripts/generate_report.py`.
