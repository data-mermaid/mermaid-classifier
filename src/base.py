import os
import spacer


class TrainingTask(object):

    def __init__(self, *args, **kwargs):
        self.overwrite = kwargs.get("overwrite") or os.environ.get("overwrite") or False

    def run(self, **kwargs):
        raise NotImplementedError("`run` must be defined")
