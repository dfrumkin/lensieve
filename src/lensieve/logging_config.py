import logging
from pathlib import Path

from lensieve.consts import logs_path


def setup_logging(root: Path, verbose: bool = False) -> None:
    log_dir = logs_path(root)
    log_dir.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO

    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

    logger = logging.getLogger()
    logger.setLevel(level)

    # avoid duplicate handlers if called twice
    logger.handlers.clear()

    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(log_dir / "lensieve.log"))

    for h in logger.handlers:
        h.setLevel(level)
        h.setFormatter(logging.Formatter(fmt))
