from mermaid_classifier.pyspacer.settings import set_env_vars_for_packages

# Set up env vars for PySpacer and MLflow.
# After this is run, we're ready to import anything which
# imports from spacer.
set_env_vars_for_packages()
