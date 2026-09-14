import logging

# These log one line per request at INFO and drown out the pipeline's output.
NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "filelock", "PIL")


def configure_logging(level: int = logging.INFO) -> None:
    """Sets up the log format for the experiment and quiets the dependencies."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def format_duration(seconds: float) -> str:
    """Formats a duration as `4.2s`, `5m12s` or `1h02m`."""
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"

    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"
