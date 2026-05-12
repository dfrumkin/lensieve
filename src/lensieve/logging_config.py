import logging
from pathlib import Path

from rich.logging import RichHandler

from lensieve.consts import logs_path


def setup_logging(root: str | Path, verbose: bool = False) -> None:
    logger = logging.getLogger()
    logger.handlers.clear()
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    # Re-enable loggers disabled by Hydra/dictConfig
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).disabled = False

    # Output to the screen
    stream_handler = RichHandler(level=level, rich_tracebacks=True)
    logger.addHandler(stream_handler)

    # Output to the log file
    log_dir = logs_path(Path(root))
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "lensieve.log", encoding="utf-8")
    file_handler.setLevel(level)
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    file_handler.setFormatter(logging.Formatter(fmt))

    logger.addHandler(file_handler)
