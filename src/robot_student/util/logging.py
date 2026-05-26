import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOGGER_NAME = "robot_student"


def configure_logging(level: int = logging.INFO, *, force: bool = False) -> None:
    logger = logging.getLogger(LOGGER_NAME)

    if force:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
