"""Patch the pydantic settings singleton for the duration of a test.

`settings` is a module-global, so an override that is not undone leaks into
every later test in the process.
"""

from contextlib import contextmanager

from mermaid_classifier.pyspacer.settings import settings


class SettingsOverride:
    """
    Override the specified Pydantic settings from a call of enable()
    until a call of disable().

    Example usage:
    override = SettingsOverride(aws_anonymous='True', aws_region='ca-central-1')
    override.enable()
    <some code that depends on the above settings>
    override.disable()

    Values are set with setattr, which bypasses pydantic validation, so each
    one must already be in the field's own type -- aws_anonymous is
    Literal['False', 'True'], and production compares it as a string.

    Some parts are from
    https://rednafi.com/python/patch-pydantic-settings-in-pytest/
    """

    def __init__(self, **kwargs):
        self.options = kwargs
        super().__init__()

    def enable(self):
        # Make a copy of the original settings
        self.original_settings = settings.model_copy()

        # Patch the settings with kwargs
        for key, val in self.options.items():
            # Raise an error if kwargs contains a nonexistent setting
            if not hasattr(settings, key):
                raise ValueError(f"Unknown setting: {key}")
            setattr(settings, key, val)

    def disable(self):
        # Restore the original settings
        settings.__dict__.update(self.original_settings.__dict__)


@contextmanager
def override_settings(**kwargs):
    """
    Override the specified Pydantic settings for the duration of the
    context manager.
    """
    override = SettingsOverride(**kwargs)
    override.enable()
    try:
        yield
    finally:
        override.disable()
