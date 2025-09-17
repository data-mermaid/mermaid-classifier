import argparse
from base import TrainingTask


class PyspacerTrainingTask(TrainingTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, **kwargs):
        print(self.overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing outputs instead of incrementing",
    )
    options = parser.parse_args()
    hii_export_task = PyspacerTrainingTask(**vars(options))
    hii_export_task.run()
