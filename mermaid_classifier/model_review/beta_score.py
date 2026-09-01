"""Score the Beta classifier pickle and write its top-1 label per point.

Runs as a standalone script inside a uv-managed ephemeral environment whose only
packages are scikit-learn 1.1.3 and numpy<2 (see ``beta_infer.py``), so it imports
nothing from ``mermaid_classifier`` and nothing outside stdlib/numpy/sklearn.

Input:  ``--features F.npz`` holding ``X`` of shape ``(n_points, dim)``.
Output: ``--out P.npz`` holding ``pred_bagf`` (n_points BA::GF strings) and
``sklearn_version`` (the caller verifies it is the pinned one).
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np

# predict_proba over the whole matrix at once would peak at n_points x n_classes.
BATCH = 8192


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--classifier", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    x = np.load(args.features, allow_pickle=False)["X"].astype(np.float32)

    # Unpickling an estimator stamped with another scikit-learn version warns; the
    # caller's version check is the real guard, so the warning adds only noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(args.classifier, "rb") as f:
            clf = pickle.load(f)
    import sklearn

    classes = np.array([str(c) for c in clf.classes_])
    # `beta_model/` also holds a same-shaped pickle whose classes are bare BA ids.
    # That one produces labels the review app cannot split into BA + growth form,
    # and the failure would otherwise surface far downstream.
    unseparated = [c for c in classes.tolist() if "::" not in c]
    if unseparated:
        raise SystemExit(
            f"{args.classifier} has {len(unseparated)} classes that are not BA::GF "
            f"(e.g. {unseparated[0]!r}); the review app needs the BA::GF classifier"
        )

    preds = np.empty(x.shape[0], dtype=object)
    for i in range(0, x.shape[0], BATCH):
        proba = clf.predict_proba(x[i : i + BATCH])
        preds[i : i + BATCH] = classes[proba.argmax(axis=1)]

    np.savez_compressed(
        args.out,
        pred_bagf=preds.astype(str),
        sklearn_version=str(sklearn.__version__),
    )
    print(f"scored {x.shape[0]} points over {len(classes)} classes -> {args.out}")


if __name__ == "__main__":
    main()
