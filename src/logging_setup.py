"""Central logging configuration.

The pipeline used to communicate via bare `print()` calls, which meant
diagnostics were invisible in production and impossible to filter. Every module
now logs through the standard library instead, so verbosity is controlled in one
place by the DOCCHAT_LOG_LEVEL environment variable.
"""

import logging
import os

_CONFIGURED = False


def setup_logging(level=None):
    """Configure root logging once per process.

    Safe to call repeatedly — only the first call installs a handler, so
    importing this from several modules will not duplicate log lines.
    """
    global _CONFIGURED

    if _CONFIGURED:
        return

    resolved = level or os.getenv("DOCCHAT_LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # These libraries are extremely chatty at INFO and drown out our own logs.
    for noisy in ("httpx", "httpcore", "openai", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name):
    """Return a module-scoped logger with logging already configured."""
    setup_logging()
    return logging.getLogger(name)
