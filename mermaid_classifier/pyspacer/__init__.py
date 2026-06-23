"""PySpacer training pipeline.

Env-var setup for PySpacer/MLflow is NOT done at import anymore (it used to call
``set_env_vars_for_packages()`` here). That import-time side effect forced every
consumer of this subpackage — including inference, which only needs
``torch_classifier.TorchMLPClassifier`` to load a trained model — to install the
training-only settings deps (``pydantic-settings``, ``psutil``).

Now env setup is an explicit step performed by the training entry points
(``TrainingRunner.__init__`` and the ``scripts/`` mains). Inference installs
(``mermaid-classifier[inference]``) can import ``torch_classifier`` with just
numpy + torch. ``set_env_vars_for_packages()`` is idempotent, so calling it from
more than one entry point is harmless.
"""
