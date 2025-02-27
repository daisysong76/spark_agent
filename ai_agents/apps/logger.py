import logging
import sys
from ai_agent.config.settings import settings

def setup_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(settings.LOG_LEVEL)  # e.g. INFO, DEBUG, etc.

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(settings.LOG_LEVEL)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Example: add file rotating handler if needed
    # from logging.handlers import RotatingFileHandler
    # file_handler = RotatingFileHandler("app.log", maxBytes=5_000_000, backupCount=5)
    # file_handler.setLevel(settings.LOG_LEVEL)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    return logger

logger = setup_logging()
