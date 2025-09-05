import sys
from loguru import logger
import os
from datetime import datetime

# Clear default handlers
logger.remove()

# Add console handler with colorized output
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
    level="DEBUG" if os.getenv("FLASK_ENV") == "development" else "INFO",
)

# Optional: also log to file (rotating daily, keep 7 days)
os.makedirs("logs", exist_ok=True)
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger.add(
    f"logs/memoir_{now}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    encoding="utf8"
)

# expose logger instance
log = logger
