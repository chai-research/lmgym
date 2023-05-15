import logging
import os
import shutil
import glob

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class CleanupCallback(TrainerCallback):
    """
    Remove files and artifacts at the end of each checkpoint save.

    For example optimizer states are often not needed but use a huge amount
    of diskspace.
    """

    def __init__(self, pattern):
        self.pattern = pattern

    def on_save(self, args, state, control, **kwargs):
        logger.info('running cleanup')

        # if using deepspeed only log for the main process
        if state.is_world_process_zero:
            files = glob.glob(self.pattern)

            if len(files) == 0:
                logger.info('found no files to cleanup')

            for p in files:
                logger.info('deleting {}'.format(p))
                self._delete_path(p)

    def _delete_path(self, path):
        if os.path.isfile(path):
            os.remote(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
