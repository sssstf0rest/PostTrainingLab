"""Console logging.

WHY THIS FILE EXISTS
--------------------
Training runs for hours and prints thousands of lines. Without timestamps and
levels, a log is unreadable when something goes wrong at 3am into a run. This is
the boring infrastructure that makes debugging possible later.

Note: this is *console* logging. Metric tracking (loss curves, LR, grad norm,
tokens/sec) is a different concern and gets W&B in M4.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "optl", level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        _CONFIGURED = True
    return logger
