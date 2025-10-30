"""
Default settings for the Einhorn package.
These settings can be overridden in your application as needed.
"""

import logging
from pathlib import Path

# Logging settings
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = Path("logs/einhorn.log")

# Model settings
DEFAULT_MODEL_ID = 1
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1.5  # seconds

# Example settings
EXAMPLE_CONFIG_PATH = Path("examples/config.json")
EXAMPLE_LOG_PATH = Path("examples/logs") 