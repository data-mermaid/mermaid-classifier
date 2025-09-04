import os
from pathlib import Path
import sys


def setup_packages():

    # In most cases, cwd() should be the dir this notebook lives in.
    # If it's not, you'll have to specify that dir using the NOTEBOOKS_DIR
    # environment var.
    examples_dir = Path(os.environ.get('NOTEBOOKS_DIR') or Path.cwd())
    # Allow importing from the core folder.
    core_dir = examples_dir.parent / 'core'
    if core_dir not in sys.path:
        sys.path.append(str(core_dir))

    # Set up our preferred way of configuring pyspacer.
    #
    # Specifically, support defining SPACER_EXTRACTORS_CACHE_DIR in an .env
    # file, where it can be defined along with our non-spacer config vars.
    from utils import Settings
    cache_dir = Settings().SPACER_EXTRACTORS_CACHE_DIR
    if cache_dir:
        os.environ['SPACER_EXTRACTORS_CACHE_DIR'] = cache_dir

    # Now, Python modules under the core folder can be imported from.
